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
from sklearn.metrics import accuracy_score, roc_auc_score
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
        label = str(row[label_key]).strip()
        if text == "" or label == "" or label.lower() == "nan":
            continue
        texts.append(text)
        labels.append(label)

    return texts, labels


def build_vocab(texts: List[str], max_vocab_size: int = 50000, min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
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


def evaluate_model(model, dataloader, device):
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_texts.extend(batch["raw_text"])

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    msps = all_probs.max(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "mean_msp": float(np.mean(msps)),
        "ece": float(expected_calibration_error(all_probs, all_labels)),
    }

    return metrics, all_labels, all_preds, all_probs, msps, all_texts


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, default="amazon")
    parser.add_argument("--eval_datasets", nargs="+", default=["amazon", "dynasent", "imdb", "semeval", "sst5", "yelp"])
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_vocab_size", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_path = os.path.join(args.data_root, args.train_dataset, "train.tsv")
    train_texts, train_labels_raw = read_tsv(train_path)

    print(f"Loaded train dataset: {args.train_dataset}", flush=True)
    print(f"Train samples: {len(train_texts)}", flush=True)

    label_set = sorted(set(train_labels_raw))
    label2id = {label: i for i, label in enumerate(label_set)}
    id2label = {i: label for label, i in label2id.items()}

    train_labels = [label2id[x] for x in train_labels_raw]

    print(f"Label mapping: {label2id}", flush=True)

    vocab = build_vocab(train_texts, max_vocab_size=args.max_vocab_size)
    print(f"Vocab size: {len(vocab)}", flush=True)

    train_dataset = TextDataset(train_texts, train_labels, vocab, args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=len(label2id),
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        pad_idx=vocab["<pad>"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_textcnn.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.6f}", flush=True)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}", flush=True)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded best model for evaluation.", flush=True)

    save_json(vocab, os.path.join(args.output_dir, "vocab.json"))
    save_json(label2id, os.path.join(args.output_dir, "label2id.json"))
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    metrics_rows = []
    all_prediction_rows = []

    id_dataset_name = args.train_dataset
    id_msps_reference = None

    for dataset_name in args.eval_datasets:
        test_path = os.path.join(args.data_root, dataset_name, "test.tsv")
        if not os.path.exists(test_path):
            print(f"Skipping missing dataset: {dataset_name}", flush=True)
            continue

        texts, raw_labels = read_tsv(test_path)

        kept_texts = []
        kept_labels = []

        def normalize_label_string(label):
            s = str(label).strip()
            if s.endswith(".0"):
                s = s[:-2]
            return s

        skipped = 0
        for text, label in zip(texts, raw_labels):
            norm_label = normalize_label_string(label)
            if norm_label in label2id:
                kept_texts.append(text)
                kept_labels.append(label2id[norm_label])
            else:
                skipped += 1

        if len(kept_texts) == 0:
            print(f"Skipping {dataset_name}: no labels matched training label space.", flush=True)
            continue

        if skipped > 0:
            print(f"{dataset_name}: skipped {skipped} unseen-label examples.", flush=True)

        eval_dataset = TextDataset(kept_texts, kept_labels, vocab, args.max_len)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        metrics, labels, preds, probs, msps, raw_texts = evaluate_model(model, eval_loader, device)

        if dataset_name == id_dataset_name:
            id_msps_reference = msps

        row = {
            "dataset": dataset_name,
            "num_samples": len(labels),
            "accuracy": metrics["accuracy"],
            "mean_msp": metrics["mean_msp"],
            "ece": metrics["ece"],
            "shift_type": "in_domain" if dataset_name == id_dataset_name else "ood"
        }

        metrics_rows.append(row)

        for i in range(len(labels)):
            pred_row = {
                "dataset": dataset_name,
                "text": raw_texts[i],
                "true_label_id": int(labels[i]),
                "pred_label_id": int(preds[i]),
                "true_label": id2label[int(labels[i])],
                "pred_label": id2label[int(preds[i])],
                "msp": float(msps[i]),
                "correct": int(labels[i] == preds[i]),
            }

            for cls_idx in range(probs.shape[1]):
                pred_row[f"prob_class_{cls_idx}"] = float(probs[i, cls_idx])

            all_prediction_rows.append(pred_row)

        print(
            f"{dataset_name}: samples={len(labels)} "
            f"acc={metrics['accuracy']:.4f} "
            f"mean_msp={metrics['mean_msp']:.4f} "
            f"ece={metrics['ece']:.4f}",
            flush=True
        )

    metrics_df = pd.DataFrame(metrics_rows)

    if id_msps_reference is not None:
        aurocs = []
        for row in metrics_rows:
            dataset_name = row["dataset"]
            if dataset_name == id_dataset_name:
                continue

            pred_df = pd.DataFrame(all_prediction_rows)
            ood_msps = pred_df[pred_df["dataset"] == dataset_name]["msp"].values

            try:
                auroc = roc_auc_score(
                    np.concatenate([np.zeros(len(id_msps_reference)), np.ones(len(ood_msps))]),
                    np.concatenate([1.0 - id_msps_reference, 1.0 - ood_msps])
                )
            except Exception:
                auroc = float("nan")

            aurocs.append({"dataset": dataset_name, "id_vs_ood_auroc_using_1_minus_msp": auroc})

        auroc_df = pd.DataFrame(aurocs)
        metrics_df = metrics_df.merge(auroc_df, on="dataset", how="left")
    else:
        metrics_df["id_vs_ood_auroc_using_1_minus_msp"] = np.nan

    metrics_csv = os.path.join(args.output_dir, "experiment4_metrics.csv")
    preds_csv = os.path.join(args.output_dir, "experiment4_predictions.csv")

    metrics_df.to_csv(metrics_csv, index=False)
    pd.DataFrame(all_prediction_rows).to_csv(preds_csv, index=False)

    summary = {
        "train_dataset": args.train_dataset,
        "eval_datasets": args.eval_datasets,
        "device": str(device),
        "best_train_loss": best_loss,
        "metrics_csv": metrics_csv,
        "predictions_csv": preds_csv,
        "model_path": best_model_path,
    }
    save_json(summary, os.path.join(args.output_dir, "summary.json"))

    print(f"Saved metrics to: {metrics_csv}", flush=True)
    print(f"Saved predictions to: {preds_csv}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
