import os
import math
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


RESULTS_ROOT = "./results/evaluations/method"
WILDTIME_RESULTS = "./wildtime_exp3/results"
OUT_DIR = "./experiment6_outputs"
MAX_ITERS = 500
LR = 0.01
N_BINS = 15
EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.clip(exp_z.sum(axis=1, keepdims=True), EPS, None)


def nll_with_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits / temperature)
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, EPS, 1.0))))


def compute_ece_from_probs(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if not np.any(mask):
            continue

        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)

    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = LR, max_iters: int = MAX_ITERS) -> float:
    log_t = 0.0

    for _ in range(max_iters):
        t = math.exp(log_t)
        scaled = logits / t
        probs = softmax(scaled)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1.0

        expected_logits = np.sum(probs * logits, axis=1)
        true_logits = logits[np.arange(len(labels)), labels]
        grad_t = np.mean((true_logits - expected_logits) / (t * t))
        grad_log_t = grad_t * t

        log_t -= lr * grad_log_t

        log_t = float(np.clip(log_t, math.log(0.05), math.log(20.0)))

    return float(math.exp(log_t))


def load_logits_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    logit_cols = sorted([c for c in df.columns if c.startswith("logit_")], key=lambda x: int(x.split("_")[1]))
    if "true_label" not in df.columns:
        raise ValueError(f"Missing true_label column in {path}")
    if not logit_cols:
        raise ValueError(f"No logit columns found in {path}")

    logits = df[logit_cols].to_numpy(dtype=np.float64)
    labels = df["true_label"].to_numpy(dtype=np.int64)
    return logits, labels


def evaluate_file(eval_path: str, temperature: float) -> Dict:
    logits, labels = load_logits_csv(eval_path)

    probs_before = softmax(logits)
    probs_after = softmax(logits / temperature)

    ece_before = compute_ece_from_probs(probs_before, labels)
    ece_after = compute_ece_from_probs(probs_after, labels)
    acc_before = float((np.argmax(probs_before, axis=1) == labels).mean())
    acc_after = float((np.argmax(probs_after, axis=1) == labels).mean())

    return {
        "file": eval_path,
        "num_examples": int(len(labels)),
        "num_classes": int(logits.shape[1]),
        "temperature": float(temperature),
        "acc_before": acc_before,
        "acc_after": acc_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_reduction": ece_before - ece_after,
    }


def get_exp2_calibration_and_eval_files() -> List[Tuple[str, str, List[str]]]:
    tasks = []

    for source in ["dynasent", "imdb", "semeval", "sst5", "yelp"]:
        base = os.path.join(
            RESULTS_ROOT, "SentimentAnalysis", source, "t5-small", "vanilla", "logits", "0"
        )
        calib = os.path.join(base, f"{source}_logits.csv")
        evals = sorted(glob.glob(os.path.join(base, "*_logits.csv")))
        if os.path.exists(calib) and evals:
            tasks.append(("exp2_sentiment", calib, evals))

    for source in ["abuse_analyzer", "civil_comments", "implicit_hate", "toxigen"]:
        base = os.path.join(
            RESULTS_ROOT, "ToxicDetection", source, "t5-small", "vanilla", "logits", "0"
        )
        calib = os.path.join(base, f"{source}_logits.csv")
        evals = sorted(glob.glob(os.path.join(base, "*_logits.csv")))
        if os.path.exists(calib) and evals:
            tasks.append(("exp2_toxic", calib, evals))

    for source in ["anli", "contract_nli", "wanli"]:
        base = os.path.join(
            RESULTS_ROOT, "NaturalLanguageInference", source, "t5-small", "vanilla", "logits", "0"
        )
        calib = os.path.join(base, f"{source}_logits.csv")
        evals = sorted(glob.glob(os.path.join(base, "*_logits.csv")))
        if os.path.exists(calib) and evals:
            tasks.append(("exp2_nli", calib, evals))

    for source in ["conll", "ener", "wnut"]:
        base = os.path.join(
            RESULTS_ROOT, "NameEntityRecognition", source, "deberta-small", "vanilla", "logits", "0"
        )
        calib = os.path.join(base, f"{source}_token_logits.csv")
        evals = sorted(glob.glob(os.path.join(base, "*_token_logits.csv")))
        if os.path.exists(calib) and evals:
            tasks.append(("exp2_ner", calib, evals))

    return tasks


def get_exp3_calibration_and_eval_files() -> List[Tuple[str, str, List[str]]]:
    tasks = []

    huff_val = os.path.join(WILDTIME_RESULTS, "exp3_huffpost_val_logits.csv")
    huff_evals = sorted(glob.glob(os.path.join(WILDTIME_RESULTS, "exp3_huffpost_*_logits.csv")))
    huff_evals = [p for p in huff_evals if not p.endswith("val_logits.csv")]
    if os.path.exists(huff_val) and huff_evals:
        tasks.append(("exp3_huffpost", huff_val, huff_evals))

    arxiv_val = os.path.join(WILDTIME_RESULTS, "exp3_arxiv_val_logits.csv")
    arxiv_evals = sorted(glob.glob(os.path.join(WILDTIME_RESULTS, "exp3_arxiv_*_logits.csv")))
    arxiv_evals = [p for p in arxiv_evals if not p.endswith("val_logits.csv")]
    if os.path.exists(arxiv_val) and arxiv_evals:
        tasks.append(("exp3_arxiv", arxiv_val, arxiv_evals))

    return tasks


def summarize_group(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["group", "calibration_file"], as_index=False)
        .agg(
            num_eval_files=("file", "count"),
            mean_temperature=("temperature", "mean"),
            mean_acc_before=("acc_before", "mean"),
            mean_acc_after=("acc_after", "mean"),
            mean_ece_before=("ece_before", "mean"),
            mean_ece_after=("ece_after", "mean"),
            mean_ece_reduction=("ece_reduction", "mean"),
        )
    )
    return grouped


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    all_tasks = get_exp2_calibration_and_eval_files() + get_exp3_calibration_and_eval_files()
    if not all_tasks:
        raise RuntimeError("No calibration/evaluation files found.")

    rows: List[Dict] = []
    temp_rows: List[Dict] = []

    for group_name, calib_path, eval_paths in all_tasks:
        calib_logits, calib_labels = load_logits_csv(calib_path)
        temperature = fit_temperature(calib_logits, calib_labels)

        temp_rows.append(
            {
                "group": group_name,
                "calibration_file": calib_path,
                "num_calibration_examples": int(len(calib_labels)),
                "num_classes": int(calib_logits.shape[1]),
                "temperature": temperature,
                "calibration_nll_before": nll_with_temperature(calib_logits, calib_labels, 1.0),
                "calibration_nll_after": nll_with_temperature(calib_logits, calib_labels, temperature),
                "calibration_ece_before": compute_ece_from_probs(softmax(calib_logits), calib_labels),
                "calibration_ece_after": compute_ece_from_probs(softmax(calib_logits / temperature), calib_labels),
            }
        )

        for eval_path in eval_paths:
            result = evaluate_file(eval_path, temperature)
            result["group"] = group_name
            result["calibration_file"] = calib_path
            rows.append(result)

    detail_df = pd.DataFrame(rows).sort_values(["group", "file"]).reset_index(drop=True)
    temp_df = pd.DataFrame(temp_rows).sort_values(["group", "calibration_file"]).reset_index(drop=True)
    summary_df = summarize_group(detail_df)

    detail_path = os.path.join(OUT_DIR, "exp6_ece_detail.csv")
    temp_path = os.path.join(OUT_DIR, "exp6_temperatures.csv")
    summary_path = os.path.join(OUT_DIR, "exp6_ece_summary.csv")

    detail_df.to_csv(detail_path, index=False)
    temp_df.to_csv(temp_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved detailed results to {detail_path}", flush=True)
    print(f"Saved temperatures to {temp_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
