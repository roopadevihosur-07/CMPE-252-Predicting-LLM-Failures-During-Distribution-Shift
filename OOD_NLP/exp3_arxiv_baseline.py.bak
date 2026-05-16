import os
import random
import numpy as np
import torch

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wild_time_data import load_dataset, num_outputs

Seed = 0
Device = "cuda" if torch.cuda.is_available() else "cpu"

train_years = [2007, 2008, 2009, 2010]
test_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

data_dir = "./wildtime_exp3/Data"
results_dir = "./wildtime_exp3/results"
ckpt_dir = "./wildtime_exp3/checkpoints"

model_name = "./model_cache/deberta-v3-small"
batch_size = 8
max_len = 192
lr = 2e-5
epochs = 2
val_frac = 0.1

os.makedirs(results_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
def saveLogitsCSV(model, dataloader, device, outCSV):
    model.eval()
    allLogits = []
    allLabels = []

    with torch.no_grad():
    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            input_ids, attention_mask, labels = batch

        else:
            first, labels = batch
            input_ids = first["input_ids"]
            attention_mask = first["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        allLogits.append(logits.detach().cpu())
        allLabels.append(labels.detach().cpu())

    logits = torch.cat(allLogits, dim=0).numpy()
    labels = torch.cat(allLabels, dim=0).numpy()

    data = {"true_label": labels.tolist()}
    for i in range(logits.shape[1]):
        data[f"logit_{i}"] = logits[:, i].tolist()

    os.makedirs(os.path.dirname(outCSV), exist_ok=True)
    pd.DataFrame(data).to_csv(outCSV, index=False)
    print(f"Saved the logits to {outCSV}", flush=True)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TextLabelDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        text, label = self.base_ds[idx]
        return str(text), int(label)

def make_concat_dataset(years, split):
    parts = []
    for y in years:
        print(f"Loading {split} split for year {y}...", flush=True)
        ds = load_dataset(
            dataset_name="arxiv",
            time_step=y,
            split=split,
            data_dir=data_dir
        )
        parts.append(TextLabelDataset(ds))
    return ConcatDataset(parts)

def split_train_val(dataset, val_frac=0.1, seed=0):
    n = len(dataset)
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_n = int(n * val_frac)
    val_idxs = idxs[:val_n]
    train_idxs = idxs[val_n:]
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)

def collate_fn(batch, tokenizer):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return enc, labels

def compute_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (conf > left) & (conf <= right) if i > 0 else (conf >= left) & (conf <= right)
        if mask.sum() == 0:
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)

def evaluate(model, loader):
    model.eval()
    all_probs = []
    allLabels = []

    with torch.no_grad():
        for enc, labels in loader:
            enc = {k: v.to(Device) for k, v in enc.items()}
            labels = labels.to(Device)

            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            allLabels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(allLabels, axis=0)

    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean() * 100.0)
    ece = compute_ece(probs, labels)
    return acc, ece

def train():
    set_seed(Seed)

    print("train_years =", train_years, flush=True)
    print("test_years =", test_years, flush=True)
    print("Loading tokenizer from local cache...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        use_fast=False
    )

    print("Building full training dataset...", flush=True)
    full_train = make_concat_dataset(train_years, split="train")
    train_ds, val_ds = split_train_val(full_train, val_frac=val_frac, seed=Seed)

    print("full_train len =", len(full_train), flush=True)
    print("train_ds len =", len(train_ds), flush=True)
    print("val_ds len =", len(val_ds), flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    num_labels = num_outputs("arxiv")
    print("num_labels =", num_labels, flush=True)

    print("Checking label range in sampled training data...", flush=True)
    min_label = 10**9
    max_label = -1
    check_n = min(len(full_train), 20000)

    for i in range(check_n):
        _, y = full_train[i]
        min_label = min(min_label, y)
        max_label = max(max_label, y)

    print(f"Observed label range in sample: min={min_label}, max={max_label}", flush=True)
    assert min_label >= 0, f"Negative label found: {min_label}"
    assert max_label < num_labels, f"Found label {max_label} but num_labels={num_labels}"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).to(Device)

    print("classifier weight shape =", model.classifier.weight.shape, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_path = os.path.join(ckpt_dir, "exp3_arxiv_best.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for enc, labels in train_loader:
            enc = {k: v.to(Device) for k, v in enc.items()}
            labels = labels.to(Device)

            out = model(**enc, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 500 == 0:
                print(f"Epoch {epoch+1}, step {steps}, avg_loss={total_loss/steps:.4f}", flush=True)

        avg_loss = total_loss / max(steps, 1)
        val_acc, val_ece = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.2f}, val_ece={val_ece:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Saved best model to {best_path}", flush=True)

    model.load_state_dict(torch.load(best_path, map_location=Device))

    saveLogitsCSV(model, val_loader, Device, "./wildtime_exp3/results/exp3_arxiv_val_logits.csv")

    out_path = os.path.join(results_dir, "exp3_arxiv_results.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dataset,train_years,test_year,time_gap,accuracy,ece\n")
        for year in test_years:
            print(f"Evaluating test year {year}", flush=True)
            test_ds = TextLabelDataset(load_dataset(
                dataset_name="arxiv",
                time_step=year,
                split="test",
                data_dir=data_dir
            ))
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, tokenizer)
            )

            acc, ece = evaluate(model, test_loader)
            gap = year - max(train_years)
            saveLogitsCSV(model, test_loader, Device, f"./wildtime_exp3/results/exp3_arxiv_{year}_logits.csv")
            print(f"test_year={year}, gap={gap}, acc={acc:.2f}, ece={ece:.4f}", flush=True)
            f.write(f"arxiv,\"{train_years}\",{year},{gap},{acc:.4f},{ece:.6f}\n")

    print(f"Saved results to {out_path}", flush=True)

if __name__ == "__main__":
    train()
