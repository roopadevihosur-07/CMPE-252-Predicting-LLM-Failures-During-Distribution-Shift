from typing import Dict
import numpy as np
import pandas as pd


def compute_accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float(df["correct"].mean())


def compute_ece_from_confidence(
    confidences_0_1: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences_0_1 >= left) & (confidences_0_1 <= right)
        else:
            mask = (confidences_0_1 >= left) & (confidences_0_1 < right)

        if not np.any(mask):
            continue

        bin_acc = correctness[mask].mean()
        bin_conf = confidences_0_1[mask].mean()
        ece += (mask.sum() / len(confidences_0_1)) * abs(bin_acc - bin_conf)

    return float(ece)


def summarize_predictions(df: pd.DataFrame) -> Dict:
    valid = df.dropna(subset=["confidence_numeric"]).copy()
    if len(valid) > 0:
        conf = (valid["confidence_numeric"].to_numpy(dtype=float) / 100.0).clip(0.0, 1.0)
        corr = valid["correct"].to_numpy(dtype=float)
        ece = compute_ece_from_confidence(conf, corr)
        mean_conf = float(valid["confidence_numeric"].mean())
    else:
        ece = float("nan")
        mean_conf = float("nan")

    return {
        "num_examples": int(len(df)),
        "accuracy": compute_accuracy(df),
        "mean_confidence": mean_conf,
        "ece": ece,
    }
