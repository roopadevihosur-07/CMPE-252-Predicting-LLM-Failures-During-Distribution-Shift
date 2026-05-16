import os
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .dataloader import *
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.utils import signature
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import random
import numpy as np
import torch


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], max_length=256, padding=True, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["tag_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["label"] = labels

    return tokenized_inputs


from collections import Counter
def sampling_ner(dataset, num_classes, shots):
    dataset = dataset.shuffle()

    count = np.zeros((num_classes,), dtype=np.int64)
    sampled_dataset = []

    for i, data in enumerate(dataset):
        if data["tag_ids"].count(0) == len(data["tag_ids"]):
            continue

        count_update = deepcopy(count)
        count_sentence = Counter(data["tag_ids"])

        required_tags = [tag_id for tag_id in range(num_classes) if count[tag_id] < shots]
        if all([count_sentence[tag_id] == 0 for tag_id in required_tags]):
            continue

        for tag_id in range(num_classes):
            count_update[tag_id] += count_sentence[tag_id]

        num_entities = [item for item in count_update[1:]]
        if max(num_entities) > 2 * shots:
            continue

        del count
        count = count_update
        sampled_dataset.append(i)

        if min(num_entities) >= shots:
            break

    print(count)
    print(len(sampled_dataset))

    return dataset.select(sampled_dataset)


def _save_token_scores_csv(save_path, rows):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"saved ner token scores csv: {save_path}", flush=True)


def evaluation(model, test_dataloader, ood_name, label_mapping, save_path=None):
    model.eval()

    references = []
    predictions = []
    token_rows = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_masks = batch["attention_mask"].cuda()
            labels = batch["label"]

            tags = [[label_mapping[l] for l in label if l != -100] for label in labels]

            logits = model(input_ids=input_ids, attention_mask=attention_masks).logits
            probs = torch.nn.functional.softmax(logits, -1)

            assert probs.shape[-1] == 9 or probs.shape[-1] == 17
            if probs.shape[-1] == 17:
                if ood_name == "conll" or ood_name == "ener":
                    probs[..., 9:] = 0
                elif ood_name == "wnut":
                    probs[..., 7:11] = 0
                    probs[..., 15:] = 0
                elif ood_name == "crossner":
                    probs[..., 9:11] = 0

            preds = probs.argmax(-1).detach().cpu().tolist()
            probs_cpu = probs.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()

            preds = [[label_mapping[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]

            # save token-level rows for calibration-style analysis
            for b_idx in range(len(labels)):
                valid_positions = [j for j, l in enumerate(labels[b_idx].tolist()) if l != -100]
                for pos_idx, j in enumerate(valid_positions):
                    true_id = int(labels[b_idx][j].item())
                    pred_id = int(np.argmax(probs_cpu[b_idx, j]))
                    row = {
                        "true_label": true_id,
                        "pred_label": pred_id,
                        "correct": int(pred_id == true_id),
                        "confidence": float(np.max(probs_cpu[b_idx, j])),
                    }
                    for c in range(logits_cpu.shape[-1]):
                        row[f"logit_{c}"] = float(logits_cpu[b_idx, j, c])
                    token_rows.append(row)

            references.extend(tags)
            predictions.extend(preds)

    metrics = datasets.load_metric("./src/utils/seqeval_metric.py", trust_remote_code=True)
    results = metrics.compute(predictions=predictions, references=references)

    if save_path is not None:
        _save_token_scores_csv(save_path, token_rows)

    model.train()
    print('f1 on {}: {}'.format(ood_name, results["overall_f1"]))

    return results["overall_precision"], results["overall_recall"], results["overall_f1"], results["overall_accuracy"]


def eval(model, processor, dataset_path, mytokenizer, result_path, task_name, ood_list, dataset_name, model_name, parameter=-1, logits_dir=None):
    print("evaluation")

    global tokenizer, soft_token_num
    tokenizer = mytokenizer
    soft_token_num = parameter
    dataset = {}
    for ood_name in ood_list:
        dataset[ood_name] = processor.get_examples(os.path.join(dataset_path, ood_name), "test")
        dataset[ood_name] = dataset[ood_name].map(tokenize_and_align_labels, fn_kwargs={"tokenizer": tokenizer}, batched=True).remove_columns(["tokens", "tags", "tag_ids"])

    dataloader_dict = {}
    for ood_name in dataset.keys():
        if os.path.exists(f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt"):
            print(f"load tokenized test dataset of {ood_name}")
            test_dataloader = torch.load(f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt", weights_only=False)
        else:
            data_collator = DataCollatorForTokenClassification(tokenizer)
            test_dataloader = DataLoader(dataset[ood_name], shuffle=False, batch_size=16, collate_fn=data_collator)
            batch_list = []
            for batch in test_dataloader:
                batch_list.append(batch)
            os.makedirs(f"./datasets/tokenize/NameEntityRecognition", exist_ok=True)
            print(f"save tokenized test dataset of {ood_name}")
            torch.save(batch_list, f"./datasets/tokenize/NameEntityRecognition/{ood_name}.pt")
            test_dataloader = batch_list

        dataloader_dict[ood_name] = test_dataloader

    print("Performance:")

    names = ["Dataset"]
    precision = ["Precision"]
    recall = ["Recall"]
    micro_f1 = ["F1"]
    accuracies = ["Acc"]
    for ood_name, test_dataloader in dataloader_dict.items():
        save_path = None
        if logits_dir is not None:
            os.makedirs(logits_dir, exist_ok=True)
            save_path = os.path.join(logits_dir, f"{ood_name}_token_logits.csv")

        p, r, f1, acc = evaluation(model, test_dataloader, ood_name, processor.labels, save_path=save_path)
        names.append(ood_name)
        precision.append(100.00 * p)
        recall.append(100.00 * r)
        micro_f1.append(100.00 * f1)
        accuracies.append(100.00 * acc)

    import pandas as pd
    results = pd.DataFrame([precision, recall, micro_f1, accuracies], columns=names)
    results.to_csv(result_path, sep="\t", index=False)

    print("finish evaluation")
