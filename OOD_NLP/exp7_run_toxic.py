import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL_ID_TO_NAME = {
    0: "non-toxic",
    1: "toxic",
}

LABEL_NAME_TO_ID = {
    "non-toxic": 0,
    "nontoxic": 0,
    "non toxic": 0,
    "benign": 0,
    "safe": 0,
    "toxic": 1,
}


def normalize_label(label: str) -> Optional[str]:
    if label is None:
        return None

    x = label.strip().lower()
    x = re.sub(r"[^a-z_ -]", "", x)
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"\s+", " ", x).strip()

    aliases = {
        "non-toxic": "non-toxic",
        "nontoxic": "non-toxic",
        "non toxic": "non-toxic",
        "benign": "non-toxic",
        "safe": "non-toxic",
        "toxic": "toxic",
    }
    return aliases.get(x)


def build_toxic_prompt(shots: List[Dict], text: str) -> str:
    if len(shots) != 4:
        raise ValueError(f"Expected exactly 4 shots, got {len(shots)}")

    sections = []
    sections.append("toxicity classification")
    sections.append("Possible labels: non-toxic, toxic.")

    for shot in shots:
        label_name = LABEL_ID_TO_NAME[int(shot["label"])]
        sections.append(
            f"Text: {shot['text']}\n"
            f"Label: {label_name}"
        )

    sections.append(
        f"Text: {text}\n"
        "Answer exactly in this format:\n"
        "Label: <non-toxic|toxic>\n"
        "Confidence: <0-100>"
    )

    return "\n\n".join(sections)


def parse_llm_response(raw_text: str) -> Tuple[Optional[str], Optional[float]]:
    if raw_text is None:
        return None, None

    text = raw_text.strip()

    label = None
    confidence = None

    m = re.search(r"Label\s*:\s*([A-Za-z _-]+)", text, flags=re.IGNORECASE)
    if m:
        label = normalize_label(m.group(1))

    if label is None:
        for candidate in ["non-toxic", "toxic"]:
            if re.search(rf"\b{candidate}\b", text, flags=re.IGNORECASE):
                label = candidate
                break
        if label is None:
            if re.search(r"\bnontoxic\b|\bnon toxic\b|\bsafe\b|\bbenign\b", text, flags=re.IGNORECASE):
                label = "non-toxic"

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


def load_toxic_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "Text" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"{path} must contain Text and Label columns")

    df = df[["Text", "Label"]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def sample_4_shots_balanced(df: pd.DataFrame, seed: int = 0) -> List[Dict]:
    shots: List[Dict] = []

    for label in [0, 1]:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            raise ValueError(f"No examples for label {label}")
        sample_n = 2
        sampled = sub.sample(n=sample_n, random_state=seed + label)
        for _, row in sampled.iterrows():
            shots.append({"text": row["text"], "label": int(row["label"])})

    return shots


def compute_ece_from_confidence(confidences_0_1, correctness, n_bins: int = 15) -> float:
    import numpy as np

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


def run_toxic_experiment(
    model_name_or_path: str,
    source_train_path: str,
    source_test_path: str,
    target_test_paths: List[str],
    output_dir: str,
    test_limit: int,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    llm = LocalHFLLM(model_name_or_path=model_name_or_path)
    source_name = os.path.basename(os.path.dirname(source_train_path))

    train_df = load_toxic_tsv(source_train_path)
    source_test_df = load_toxic_tsv(source_test_path)
    shots = sample_4_shots_balanced(train_df, seed=seed)

    eval_sets = [(source_name, source_test_df, "in_domain")]
    for target_path in target_test_paths:
        target_name = os.path.basename(os.path.dirname(target_path))
        target_df = load_toxic_tsv(target_path)
        shift_type = "in_domain" if target_name == source_name else "domain_shift"
        eval_sets.append((target_name, target_df, shift_type))

    all_rows = []
    summary_rows = []

    for target_name, eval_df, shift_type in eval_sets:
        if test_limit > 0:
            eval_df = eval_df.iloc[:test_limit].copy()

        dataset_rows = []

        for idx, row in eval_df.reset_index(drop=True).iterrows():
            prompt = build_toxic_prompt(shots=shots, text=row["text"])
            raw_response = llm.generate(prompt)
            pred_label_name, confidence = parse_llm_response(raw_response)

            pred_label_id = LABEL_NAME_TO_ID[pred_label_name] if pred_label_name is not None else None
            correct = int(pred_label_id == int(row["label"])) if pred_label_id is not None else 0

            rec = {
                "model_name": model_name_or_path,
                "task": "toxic_detection",
                "source_dataset": source_name,
                "target_dataset": target_name,
                "shift_type": shift_type,
                "example_id": idx,
                "text": row["text"],
                "gold_label": int(row["label"]),
                "gold_label_name": LABEL_ID_TO_NAME[int(row["label"])],
                "predicted_label": pred_label_id,
                "predicted_label_name": pred_label_name,
                "confidence_numeric": confidence,
                "correct": correct,
                "raw_response": raw_response,
            }
            dataset_rows.append(rec)
            all_rows.append(rec)

        dataset_df = pd.DataFrame(dataset_rows)
        metrics = summarize_predictions(dataset_df)
        summary_rows.append(
            {
                "model_name": model_name_or_path,
                "task": "toxic_detection",
                "source_dataset": source_name,
                "target_dataset": target_name,
                "shift_type": shift_type,
                **metrics,
            }
        )

    pred_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    pred_path = os.path.join(output_dir, f"exp7_toxic_predictions_{source_name}.csv")
    summary_path = os.path.join(output_dir, f"exp7_toxic_summary_{source_name}.csv")

    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved predictions to {pred_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--source_train_path", type=str, required=True)
    parser.add_argument("--source_test_path", type=str, required=True)
    parser.add_argument("--target_test_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./experiment7_outputs")
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_toxic_experiment(
        model_name_or_path=args.model_name_or_path,
        source_train_path=args.source_train_path,
        source_test_path=args.source_test_path,
        target_test_paths=args.target_test_paths,
        output_dir=args.output_dir,
        test_limit=args.test_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
