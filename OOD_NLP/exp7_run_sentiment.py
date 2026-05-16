import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp7_prompting import (
    label_id_to_name,
    label_name_to_id,
    build_sentiment_prompt,
    parse_llm_response,
)
from exp7_datasets import (
    load_sentiment_tsv,
    sample_4_shots_balanced,
    build_eval_records,
)
from exp7_metrics import summarize_predictions


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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
        return generated.strip()


def run_sentiment_experiment(
    model_name_or_path: str,
    source_train_path: str,
    source_test_path: str,
    target_test_paths: List[str],
    output_dir: str,
    test_limit: int,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    source_name = os.path.basename(os.path.dirname(source_train_path))
    llm = LocalHFLLM(model_name_or_path=model_name_or_path)

    train_df = load_sentiment_tsv(source_train_path)
    source_test_df = load_sentiment_tsv(source_test_path)

    shots = sample_4_shots_balanced(train_df, seed=seed)

    all_eval_sets = [(source_name, source_test_df, "in_domain")]
    for target_path in target_test_paths:
        target_name = os.path.basename(os.path.dirname(target_path))
        target_df = load_sentiment_tsv(target_path)
        shift_type = "in_domain" if target_name == source_name else "domain_shift"
        all_eval_sets.append((target_name, target_df, shift_type))

    all_rows = []
    summary_rows = []

    for target_name, eval_df, shift_type in all_eval_sets:
        records = build_eval_records(
            eval_df,
            source_name=source_name,
            target_name=target_name,
            shift_type=shift_type,
            limit=test_limit,
        )

        for rec in records:
            prompt = build_sentiment_prompt(shots=shots, text=rec["text"])
            raw_response = llm.generate(prompt)
            pred_label_name, confidence = parse_llm_response(raw_response)

            if pred_label_name is None:
                raw_lower = raw_response.lower()
                for candidate in ["negative", "neutral", "positive"]:
                    if candidate in raw_lower:
                        pred_label_name = candidate
                        break

            if confidence is None:
                import re
                m = re.search(r'\b([0-9]{1,3})\b', raw_response)
                if m:
                    confidence = float(max(0, min(100, int(m.group(1)))))

            pred_label_id = label_name_to_id[pred_label_name] if pred_label_name is not None else None
            correct = int(pred_label_id == rec["gold_label"]) if pred_label_id is not None else 0

            row = {
                "model_name": model_name_or_path,
                "task": "sentiment",
                "source_dataset": rec["source_dataset"],
                "target_dataset": rec["target_dataset"],
                "shift_type": rec["shift_type"],
                "example_id": rec["example_id"],
                "text": rec["text"],
                "gold_label": rec["gold_label"],
                "gold_label_name": label_id_to_name[rec["gold_label"]],
                "predicted_label": pred_label_id,
                "predicted_label_name": pred_label_name,
                "confidence_numeric": confidence,
                "correct": correct,
                "raw_response": raw_response,
            }
            all_rows.append(row)

        target_df_out = pd.DataFrame([r for r in all_rows if r["target_dataset"] == target_name])
        metrics = summarize_predictions(target_df_out)
        summary_rows.append(
            {
                "model_name": model_name_or_path,
                "task": "sentiment",
                "source_dataset": source_name,
                "target_dataset": target_name,
                "shift_type": shift_type,
                **metrics,
            }
        )

    pred_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    pred_path = os.path.join(output_dir, f"exp7_sentiment_predictions_{source_name}.csv")
    summary_path = os.path.join(output_dir, f"exp7_sentiment_summary_{source_name}.csv")

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

    run_sentiment_experiment(
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
