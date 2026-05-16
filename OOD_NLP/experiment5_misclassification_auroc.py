import os
import csv
import json
import math
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = str(text).strip()
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return normalize_text(text).lower().split()


def normalize_label_string(label):
    s = str(label).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def detect_columns(columns: List[str]) -> Tuple[str, str]:
    original_cols = [c.strip() for c in columns]
    lowered = {c.strip().lower(): c.strip() for c in columns}

    text_key = None
    label_key = None

    for candidate in ["text", "sentence", "content", "review", "tweet"]:
        if candidate in lowered:
            text_key = lowered[candidate]
            break

    for candidate in ["label", "labels", "sentiment", "class"]:
        if candidate in lowered:
            label_key = lowered[candidate]
            break

    if text_key is None or label_key is None:
        raise ValueError(f"Could not detect text/label columns. Found: {original_cols}")

    return text_key, label_key


def read_tsv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, sep="\t")
    text_key, label_key = detect_columns(df.columns.tolist())

    texts = []
    labels = []

    for _, row in df.iterrows():
        text = normalize_text(row[text_key])
        label = normalize_label_string(row[label_key])
        if text == "" or label == "" or label.lower() == "nan":
            continue
        texts.append(text)
        labels.append(label)

    return texts, labels


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        label = self.labels[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": self.texts[idx]
        }


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
             for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def extract_features(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)
        return x

    def forward(self, input_ids):
        feats = self.extract_features(input_ids)
        feats = self.dropout(feats)
        logits = self.fc(feats)
        return logits


def compute_msp_scores(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(probs, axis=1)


def compute_energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits_t = logits / temperature
    logsumexp = np.log(np.sum(np.exp(logits_t - logits_t.max(axis=1, keepdims=True)), axis=1)) + logits_t.max(axis=1)
    energy = -temperature * logsumexp
    return -energy


def fit_mahalanobis_stats(model, dataloader, device, num_classes: int):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            feats = model.extract_features(input_ids)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    class_means = []
    for c in range(num_classes):
        class_feats = all_features[all_labels == c]
        if len(class_feats) == 0:
            raise ValueError(f"No training features found for class {c}")
        class_means.append(class_feats.mean(axis=0))
    class_means = np.stack(class_means, axis=0)

    centered = []
    for c in range(num_classes):
        class_feats = all_features[all_labels == c]
        centered.append(class_feats - class_means[c])
    centered = np.vstack(centered)

    cov = EmpiricalCovariance(assume_centered=True)
    cov.fit(centered)
    precision = cov.precision_

    return class_means, precision


def compute_mahalanobis_scores(features: np.ndarray, class_means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    distances = []
    for c in range(class_means.shape[0]):
        delta = features - class_means[c]
        dist = np.einsum("bi,ij,bj->b", delta, precision, delta)
        distances.append(dist)
    distances = np.stack(distances, axis=1)
    min_dist = np.min(distances, axis=1)
    return min_dist

def collect_outputs(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []
    all_features = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            feats = model.extract_features(input_ids)
            logits = model.fc(model.dropout(feats))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_features.append(feats.cpu().numpy())
            all_texts.extend(batch["raw_text"])

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.vstack(all_probs),
        "logits": np.vstack(all_logits),
        "features": np.vstack(all_features),
        "texts": all_texts,
    }


def compute_auroc_for_misclassification(scores: np.ndarray, labels: np.ndarray, preds: np.ndarray):
    misclassified = (preds != labels).astype(int)

    if len(np.unique(misclassified)) < 2:
        return float("nan")

    return float(roc_auc_score(misclassified, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, default="amazon")
    parser.add_argument("--ood_datasets", nargs="+", default=["dynasent", "imdb", "semeval", "sst5", "yelp"])

    parser.add_argument("--model_path", type=str, default="./experiment4_cnn_outputs/best_textcnn.pt")
    parser.add_argument("--vocab_path", type=str, default="./experiment4_cnn_outputs/vocab.json")
    parser.add_argument("--label2id_path", type=str, default="./experiment4_cnn_outputs/label2id.json")
    parser.add_argument("--output_dir", type=str, default="./experiment5_outputs")

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(args.label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=len(label2id),
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        pad_idx=vocab["<pad>"]
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {args.model_path}", flush=True)

    train_path = os.path.join(args.data_root, args.train_dataset, "train.tsv")
    train_texts_raw, train_labels_raw = read_tsv(train_path)

    train_texts = []
    train_labels = []
    skipped_train = 0
    for text, label in zip(train_texts_raw, train_labels_raw):
        if label in label2id:
            train_texts.append(text)
            train_labels.append(label2id[label])
        else:
            skipped_train += 1

    if skipped_train > 0:
        print(f"Skipped {skipped_train} training examples with unmatched labels.", flush=True)

    train_dataset = TextDataset(train_texts, train_labels, vocab, args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("Fitting Mahalanobis statistics from training features...", flush=True)
    class_means, precision = fit_mahalanobis_stats(model, train_loader, device, num_classes=len(label2id))
    print("Finished Mahalanobis fitting.", flush=True)

    summary_rows = []
    prediction_rows = []

    for dataset_name in args.ood_datasets:
        test_path = os.path.join(args.data_root, dataset_name, "test.tsv")
        if not os.path.exists(test_path):
            print(f"Skipping missing dataset: {dataset_name}", flush=True)
            continue

        texts_raw, labels_raw = read_tsv(test_path)

        texts = []
        labels = []
        skipped = 0
        for text, label in zip(texts_raw, labels_raw):
            if label in label2id:
                texts.append(text)
                labels.append(label2id[label])
            else:
                skipped += 1

        if len(texts) == 0:
            print(f"Skipping {dataset_name}: no matched labels.", flush=True)
            continue

        if skipped > 0:
            print(f"{dataset_name}: skipped {skipped} unmatched-label examples.", flush=True)

        ds = TextDataset(texts, labels, vocab, args.max_len)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        outputs = collect_outputs(model, loader, device)

        msp_scores = compute_msp_scores(outputs["probs"])
        energy_scores = compute_energy_scores(outputs["logits"], temperature=args.energy_temperature)
        mahal_scores = compute_mahalanobis_scores(outputs["features"], class_means, precision)

        msp_auroc = compute_auroc_for_misclassification(msp_scores, outputs["labels"], outputs["preds"])
        energy_auroc = compute_auroc_for_misclassification(energy_scores, outputs["labels"], outputs["preds"])
        mahal_auroc = compute_auroc_for_misclassification(mahal_scores, outputs["labels"], outputs["preds"])

        row = {
            "dataset": dataset_name,
            "num_samples": len(outputs["labels"]),
            "accuracy": float(np.mean(outputs["preds"] == outputs["labels"])),
            "msp_auroc": msp_auroc,
            "mahalanobis_auroc": mahal_auroc,
            "energy_auroc": energy_auroc,
        }

        best_method = max(
            [("MSP", msp_auroc), ("Mahalanobis", mahal_auroc), ("Energy", energy_auroc)],
            key=lambda x: (-1 if math.isnan(x[1]) else x[1])
        )[0]
        row["best_method"] = best_method

        summary_rows.append(row)

        misclassified = (outputs["preds"] != outputs["labels"]).astype(int)

        for i in range(len(outputs["labels"])):
            prediction_rows.append({
                "dataset": dataset_name,
                "text": outputs["texts"][i],
                "true_label_id": int(outputs["labels"][i]),
                "pred_label_id": int(outputs["preds"][i]),
                "true_label": id2label[int(outputs["labels"][i])],
                "pred_label": id2label[int(outputs["preds"][i])],
                "misclassified": int(misclassified[i]),
                "msp_score": float(msp_scores[i]),
                "mahalanobis_score": float(mahal_scores[i]),
                "energy_score": float(energy_scores[i]),
            })

        print(
            f"[{dataset_name}] "
            f"samples={len(outputs['labels'])} "
            f"acc={row['accuracy']:.4f} "
            f"MSP_AUROC={msp_auroc:.4f} "
            f"Mahalanobis_AUROC={mahal_auroc:.4f} "
            f"Energy_AUROC={energy_auroc:.4f}",
            flush=True
        )

    summary_df = pd.DataFrame(summary_rows)
    preds_df = pd.DataFrame(prediction_rows)

    summary_csv = os.path.join(args.output_dir, "experiment5_method_comparison.csv")
    preds_csv = os.path.join(args.output_dir, "experiment5_prediction_scores.csv")

    summary_df.to_csv(summary_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)

    if len(summary_df) > 0:
        avg_row = {
            "dataset": "AVERAGE",
            "num_samples": int(summary_df["num_samples"].sum()),
            "accuracy": float(summary_df["accuracy"].mean()),
            "msp_auroc": float(summary_df["msp_auroc"].mean()),
            "mahalanobis_auroc": float(summary_df["mahalanobis_auroc"].mean()),
            "energy_auroc": float(summary_df["energy_auroc"].mean()),
            "best_method": ""
        }
        summary_df_with_avg = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
        summary_df_with_avg.to_csv(os.path.join(args.output_dir, "experiment5_method_comparison_with_average.csv"), index=False)

    print(f"Saved summary to: {summary_csv}", flush=True)
    print(f"Saved per-example scores to: {preds_csv}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
