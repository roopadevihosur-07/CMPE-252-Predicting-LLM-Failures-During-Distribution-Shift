import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wild_time_data import load_dataset


def normalize_label_text(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def parse_llm_response(raw_text: str, allowed_labels: List[str]) -> Tuple[Optional[str], Optional[float]]:
    if raw_text is None:
        return None, None

    text = raw_text.strip()

    label = None
    confidence = None

    m = re.search(r"Label\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        cand = normalize_label_text(m.group(1))
        for lab in allowed_labels:
            if cand == normalize_label_text(lab):
                label = lab
                break

    if label is None:
        lowered = normalize_label_text(text)
        for lab in allowed_labels:
            if re.search(rf"\b{re.escape(normalize_label_text(lab))}\b", lowered):
                label = lab
                break

    m = re.search(r"Confidence\s*:\s*([0-9]{1,3})", text, flags=re.IGNORECASE)
    if m:
        confidence = float(m.group(1))
    else:
        m = re.search(r"\b([0-9]{1,3})\s*%\b", text)
        if m:
            confidence = float(m.group(1))
        else:
            m = re.search(r"\b([0-9]{1,3})\b", text)
            if m:
                confidence = float(m.group(1))

    if confidence is not None:
        confidence = max(0.0, min(100.0, confidence))

    return label, confidence


def compute_ece_from_confidence(confidences_0_1, correctness, n_bins: int = 15) -> float:
    confidences_0_1 = np.asarray(confidences_0_1, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences_0_1 >= left) & (confidences_0_1 <= right)
        else:
            mask = (confidences_0_1 >= left) & (confidences_0_1 < right)

        if not mask.any():
            continue

        bin_acc = correctness[mask].mean()
        bin_conf = confidences_0_1[mask].mean()
        ece += (mask.sum() / len(confidences_0_1)) * abs(bin_acc - bin_conf)

    return float(ece)


def summarize_predictions(df: pd.DataFrame) -> Dict:
    acc = float(df["correct"].mean()) if len(df) else float("nan")

    valid = df.dropna(subset=["confidence_numeric"]).copy()
    if len(valid):
        conf = (valid["confidence_numeric"].astype(float) / 100.0).clip(0.0, 1.0)
        corr = valid["correct"].astype(float)
        ece = compute_ece_from_confidence(conf, corr)
        mean_conf = float(valid["confidence_numeric"].mean())
        confidence_coverage = float(len(valid) / len(df))
    else:
        ece = float("nan")
        mean_conf = float("nan")
        confidence_coverage = 0.0

    return {
        "num_examples": int(len(df)),
        "accuracy": acc,
        "mean_confidence": mean_conf,
        "ece": ece,
        "confidence_coverage": confidence_coverage,
    }


class LocalHFLLM:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ) -> None:
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
        return generated.strip()


def get_text_and_label(example):
    if isinstance(example, dict):
        label = example.get("y", example.get("label"))
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "headline" in example:
            text = example["headline"]
        elif "title" in example:
            text = example["title"]
        else:
            text = str(example)
        return str(text), int(label)

    if isinstance(example, tuple) and len(example) >= 2:
        return str(example[0]), int(example[1])

    raise ValueError(f"Unsupported example format: {type(example)}")


def build_prompt(task_name: str, shots: List[Dict], text: str, allowed_labels: List[str]) -> str:
    label_line = ", ".join(allowed_labels)

    sections = []
    sections.append(f"{task_name} classification")
    sections.append(f"Possible labels: {label_line}.")

    for shot in shots:
        sections.append(
            f"Text: {shot['text']}\n"
            f"Label: {shot['label_name']}"
        )

    sections.append(
        f"Text: {text}\n"
        "Answer exactly in this format:\n"
        f"Label: <one of: {label_line}>\n"
        "Confidence: <0-100>"
    )

    return "\n\n".join(sections)


def sample_4_shots_from_years(dataset_name: str, train_years: List[int], seed: int = 0) -> Tuple[List[Dict], List[str]]:
    rng = np.random.default_rng(seed)

    all_examples = []
    all_labels = set()

    for year in train_years:
        ds = load_dataset(dataset_name=dataset_name, time_step=year, split="train", data_dir="./wildtime_exp3/Data")
        for i in range(len(ds)):
            text, label = get_text_and_label(ds[i])
            all_examples.append({"text": text, "label": int(label), "year": year})
            all_labels.add(int(label))

    all_labels = sorted(all_labels)
    label_names = [str(x) for x in all_labels]

    shots = []
    for label in all_labels:
        candidates = [ex for ex in all_examples if ex["label"] == label]
        if not candidates:
            continue
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        shots.append(
            {
                "text": chosen["text"],
                "label": chosen["label"],
                "label_name": str(chosen["label"]),
            }
        )
        if len(shots) == 4:
            break

    while len(shots) < 4:
        chosen = all_examples[int(rng.integers(0, len(all_examples)))]
        shots.append(
            {
                "text": chosen["text"],
                "label": chosen["label"],
                "label_name": str(chosen["label"]),
            }
        )

    return shots[:4], label_names


def run_wildtime_experiment(
    model_name_or_path: str,
    dataset_name: str,
    train_years: List[int],
    test_years: List[int],
    output_dir: str,
    test_limit: int,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    llm = LocalHFLLM(model_name_or_path=model_name_or_path)
    shots, allowed_labels = sample_4_shots_from_years(dataset_name, train_years, seed=seed)

    all_rows = []
    summary_rows = []

    for year in test_years:
        ds = load_dataset(dataset_name=dataset_name, time_step=year, split="test", data_dir="./wildtime_exp3/Data")

        rows = []
        n = min(test_limit, len(ds)) if test_limit > 0 else len(ds)

        for idx in range(n):
            text, gold_label = get_text_and_label(ds[idx])

            prompt = build_prompt(
                task_name=dataset_name,
                shots=shots,
                text=text,
                allowed_labels=allowed_labels,
            )
            raw_response = llm.generate(prompt)
            pred_label_name, confidence = parse_llm_response(raw_response, allowed_labels=allowed_labels)

            pred_label = int(pred_label_name) if pred_label_name is not None and pred_label_name.isdigit() else None
            correct = int(pred_label == gold_label) if pred_label is not None else 0

            rec = {
                "model_name": model_name_or_path,
                "task": dataset_name,
                "train_years": str(train_years),
                "test_year": year,
                "time_gap": year - max(train_years),
                "example_id": idx,
                "text": text,
                "gold_label": gold_label,
                "gold_label_name": str(gold_label),
                "predicted_label": pred_label,
                "predicted_label_name": pred_label_name,
                "confidence_numeric": confidence,
                "correct": correct,
                "raw_response": raw_response,
            }
            rows.append(rec)
            all_rows.append(rec)

        year_df = pd.DataFrame(rows)
        metrics = summarize_predictions(year_df)
        summary_rows.append(
            {
                "model_name": model_name_or_path,
                "task": dataset_name,
                "train_years": str(train_years),
                "test_year": year,
                "time_gap": year - max(train_years),
                **metrics,
            }
        )

    pred_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    pred_path = os.path.join(output_dir, f"exp7_{dataset_name}_predictions.csv")
    summary_path = os.path.join(output_dir, f"exp7_{dataset_name}_summary.csv")

    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved predictions to {pred_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


def parse_years(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["huffpost", "arxiv"])
    parser.add_argument("--train_years", type=str, required=True)
    parser.add_argument("--test_years", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiment7_outputs")
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_wildtime_experiment(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        train_years=parse_years(args.train_years),
        test_years=parse_years(args.test_years),
        output_dir=args.output_dir,
        test_limit=args.test_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
