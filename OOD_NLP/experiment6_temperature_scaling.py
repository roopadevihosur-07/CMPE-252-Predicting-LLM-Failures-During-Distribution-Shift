import os
import json
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expected_calibration_error_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences >= start) & (confidences < end)

        if np.sum(mask) == 0:
            continue

        bin_acc = np.mean(accuracies[mask])
        bin_conf = np.mean(confidences[mask])
        ece += (np.sum(mask) / len(labels)) * abs(bin_acc - bin_conf)

    return float(ece)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def infer_logits_from_df(df: pd.DataFrame) -> np.ndarray:
    # Case 1: explicit logit_0, logit_1, ...
    logit_cols = [c for c in df.columns if c.startswith("logit_")]
    if len(logit_cols) > 0:
        logit_cols = sorted(logit_cols, key=lambda x: int(x.split("_")[1]))
        return df[logit_cols].values.astype(np.float32)

    # Case 2: single JSON-like logits column
    if "logits" in df.columns:
        logits = []
        for item in df["logits"].tolist():
            if isinstance(item, str):
                logits.append(np.array(json.loads(item), dtype=np.float32))
            else:
                logits.append(np.array(item, dtype=np.float32))
        return np.stack(logits, axis=0)

    raise ValueError(
        "Could not find logits. Expected either columns like logit_0, logit_1, ... "
        "or a 'logits' column."
    )


def infer_labels_from_df(df: pd.DataFrame) -> np.ndarray:
    label_candidates = ["true_label", "label", "labels", "y_true"]
    for col in label_candidates:
        if col in df.columns:
            return df[col].astype(int).values
    raise ValueError(
        "Could not find labels. Expected one of: true_label, label, labels, y_true"
    )


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=1e-6)
        return logits / temp

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.data = torch.clamp(self.temperature.data, min=1e-6)

        return float(self.temperature.item())

    def transform(self, logits: np.ndarray) -> np.ndarray:
        temp = max(float(self.temperature.item()), 1e-6)
        return logits / temp


def evaluate_ece(logits: np.ndarray, labels: np.ndarray, n_bins: int) -> Tuple[float, np.ndarray]:
    probs = softmax_numpy(logits)
    ece = expected_calibration_error_from_probs(probs, labels, n_bins=n_bins)
    return ece, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_csv", type=str, required=True,
                        help="In-domain validation CSV containing true labels and logits")
    parser.add_argument("--target_csvs", nargs="+", required=True,
                        help="One or more target CSVs from Experiments 2 and 3")
    parser.add_argument("--target_names", nargs="+", default=None,
                        help="Optional names matching target_csvs order")
    parser.add_argument("--output_dir", type=str, default="./experiment6_outputs")
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--lbfgs_max_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.target_names is not None and len(args.target_names) != len(args.target_csvs):
        raise ValueError("--target_names must have the same length as --target_csvs")

    # -----------------------------
    # Load validation set
    # -----------------------------
    val_df = pd.read_csv(args.validation_csv)
    val_logits = infer_logits_from_df(val_df)
    val_labels = infer_labels_from_df(val_df)

    # -----------------------------
    # Fit temperature
    # -----------------------------
    scaler = TemperatureScaler(init_temp=1.0)

    val_ece_before, val_probs_before = evaluate_ece(val_logits, val_labels, n_bins=args.n_bins)
    learned_temp = scaler.fit(val_logits, val_labels, max_iter=args.lbfgs_max_iter)
    val_logits_scaled = scaler.transform(val_logits)
    val_ece_after, val_probs_after = evaluate_ece(val_logits_scaled, val_labels, n_bins=args.n_bins)

    print(f"Learned temperature T = {learned_temp:.6f}", flush=True)
    print(f"Validation ECE before = {val_ece_before:.6f}", flush=True)
    print(f"Validation ECE after  = {val_ece_after:.6f}", flush=True)
    print(f"Validation ECE reduction = {val_ece_before - val_ece_after:.6f}", flush=True)

    # Save temperature
    with open(os.path.join(args.output_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"temperature": learned_temp}, f, indent=2)

    # -----------------------------
    # Evaluate target distributions
    # -----------------------------
    rows = []

    for i, target_csv in enumerate(args.target_csvs):
        name = args.target_names[i] if args.target_names is not None else os.path.splitext(os.path.basename(target_csv))[0]

        df = pd.read_csv(target_csv)
        logits = infer_logits_from_df(df)
        labels = infer_labels_from_df(df)

        ece_before, probs_before = evaluate_ece(logits, labels, n_bins=args.n_bins)

        scaled_logits = scaler.transform(logits)
        ece_after, probs_after = evaluate_ece(scaled_logits, labels, n_bins=args.n_bins)

        reduction = ece_before - ece_after

        row = {
            "distribution": name,
            "num_samples": len(labels),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_reduction": reduction,
            "temperature": learned_temp,
        }
        rows.append(row)

        print(
            f"[{name}] samples={len(labels)} "
            f"ECE_before={ece_before:.6f} "
            f"ECE_after={ece_after:.6f} "
            f"reduction={reduction:.6f}",
            flush=True
        )

        # save per-example confidence before/after
        out_df = pd.DataFrame({
            "true_label": labels,
            "pred_before": probs_before.argmax(axis=1),
            "conf_before": probs_before.max(axis=1),
            "pred_after": probs_after.argmax(axis=1),
            "conf_after": probs_after.max(axis=1),
            "correct_before": (probs_before.argmax(axis=1) == labels).astype(int),
            "correct_after": (probs_after.argmax(axis=1) == labels).astype(int),
        })
        out_df.to_csv(os.path.join(args.output_dir, f"{name}_confidence_before_after.csv"), index=False)

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(args.output_dir, "experiment6_ece_reduction.csv")
    results_df.to_csv(results_csv, index=False)

    if len(results_df) > 0:
        avg_row = {
            "distribution": "AVERAGE",
            "num_samples": int(results_df["num_samples"].sum()),
            "ece_before": float(results_df["ece_before"].mean()),
            "ece_after": float(results_df["ece_after"].mean()),
            "ece_reduction": float(results_df["ece_reduction"].mean()),
            "temperature": learned_temp,
        }
        results_with_avg = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        results_with_avg.to_csv(
            os.path.join(args.output_dir, "experiment6_ece_reduction_with_average.csv"),
            index=False
        )

    print(f"Saved Experiment 6 summary to: {results_csv}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
