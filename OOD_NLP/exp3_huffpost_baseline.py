import os
import math
import random
import numpy as np
import torch

import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from wild_time_data import load_dataset

Seed = 0
Device = "cuda" if torch.cuda.is_available() else "cpu"

Train_Years = [2012, 2013, 2014, 2015]
Test_Years = [2016, 2017, 2018]

Data_Dir = "./wildtime_exp3/Data"
Results_Dir = "./wildtime_exp3/results"
Ckpt_Dir = "./wildtime_exp3/checkpoints"

Model_Name = "./model_cache/deberta-v3-small"
Batch_Size = 16
Max_Len = 256
LR = 2e-5
Epochs = 3
Val_Frac = 0.1

os.makedirs(Results_Dir, exist_ok=True)
os.makedirs(Ckpt_Dir, exist_ok=True)

def save_logits_csv(model, dataloader, device, out_csv):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                elif len(batch) == 2:
                    first, second = batch

                    if hasattr(first, "keys") and "input_ids" in first and "attention_mask" in first:
                        input_ids = first["input_ids"].to(device)
                        attention_mask = first["attention_mask"].to(device)
                        labels = second.to(device)
                    else:
                        raise ValueError(f"This is an unsupported 2-item batch format: first element type = {type(first)}")

                else:
                    raise ValueError(f"This is an unsupported batch length: {len(batch)}")

            else:
                raise ValueError(f"This is an unsupported batch type: {type(batch)}")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    data = {"true_label": labels.tolist()}
    for i in range(logits.shape[1]):
        data[f"logit_{i}"] = logits[:, i].tolist()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False)
    print(f"Saved logits to {out_csv}", flush=True)


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
        ds = load_dataset(dataset_name="huffpost", time_step=y,split=split,data_dir=Data_Dir)
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
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=Max_Len, return_tensors="pt")
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
    all_labels = []

    with torch.no_grad():
        for enc, labels in loader:
            enc = {k: v.to(Device) for k, v in enc.items()}
            labels = labels.to(Device)

            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean() * 100.0)
    ece = compute_ece(probs, labels)
    return acc, ece

def train():
    set_seed(Seed)

    tokenizer = AutoTokenizer.from_pretrained(Model_Name)

    full_train = make_concat_dataset(Train_Years, split="train")
    train_ds, val_ds = split_train_val(full_train, val_frac=Val_Frac, seed=Seed)

    train_loader = DataLoader(train_ds, batch_size=Batch_Size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=Batch_Size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    sample_ds = load_dataset(dataset_name="huffpost", time_step=2012, split="train", data_dir=Data_Dir)
    labels = set(int(sample_ds[i][1]) for i in range(min(len(sample_ds), 5000)))
    num_labels = max(labels) + 1
    model = AutoModelForSequenceClassification.from_pretrained(Model_Name, num_labels=num_labels, local_files_only=True, ignore_mismatched_sizes=True).to(Device)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_path = os.path.join(Ckpt_Dir, "exp3_huffpost_best.pt")

    for epoch in range(Epochs):
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

        avg_loss = total_loss / max(steps, 1)
        val_acc, val_ece = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.2f}, val_ece={val_ece:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Saved best model to {best_path}", flush=True)

    model.load_state_dict(torch.load(best_path, map_location=Device))

    save_logits_csv(model, val_loader, Device, "./wildtime_exp3/results/exp3_huffpost_val_logits.csv")

    out_path = os.path.join(Results_Dir, "exp3_huffpost_results.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dataset,train_years,test_year,time_gap,accuracy,ece\n")
        for year in Test_Years:
            test_ds = TextLabelDataset(load_dataset(dataset_name="huffpost", time_step=year, split="test", data_dir=Data_Dir))
            test_loader = DataLoader(test_ds, batch_size=Batch_Size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

            acc, ece = evaluate(model, test_loader)
            gap = year - max(Train_Years)
            save_logits_csv(model, test_loader, Device, f"./wildtime_exp3/results/exp3_huffpost_{year}_logits.csv")
            print(f"test_year={year}, gap={gap}, acc={acc:.2f}, ece={ece:.4f}", flush=True)
            f.write(f"huffpost,\"{Train_Years}\",{year},{gap},{acc:.4f},{ece:.6f}\n")

    print(f"Saved results to {out_path}", flush=True)

if __name__ == "__main__":
    train()
