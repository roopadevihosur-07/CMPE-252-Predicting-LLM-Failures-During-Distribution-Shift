import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .dataloader import *
from openprompt import PromptDataLoader
from tqdm import tqdm

import random
import numpy as np
import torch


def sampling(dataset, shots):
    subsets = {}
    sampled_dataset = []
    for data in dataset:
        if data.label not in subsets.keys():
            subsets[data.label] = []
        subsets[data.label].append(data)

    for label, subset in subsets.items():
        random.shuffle(subset)
        shots = min(shots, len(subset))
        sampled_dataset.extend(subset[:shots])

    random.shuffle(sampled_dataset)

    return sampled_dataset


def _save_scores_csv(save_path, labels, preds, probs, logits=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "true_label": labels,
        "pred_label": preds,
        "correct": [int(i == j) for i, j in zip(preds, labels)],
        "confidence": [float(np.max(p)) for p in probs],
    }

    if logits is not None:
        for i in range(logits.shape[1]):
            data[f"logit_{i}"] = logits[:, i].tolist()
    else:
        for i in range(probs.shape[1]):
            data[f"prob_{i}"] = probs[:, i].tolist()

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"saved scores/logits csv: {save_path}", flush=True)


def evaluation(test_dataloader, prompt_model, task_name, dataset_name, model_name, ood_name, save_path=None):
    prompt_model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            probs = F.softmax(logits, dim=-1)

            save_logits = logits
            save_probs = probs

            if task_name == "SentimentAnalysis" and ood_name in ["dsc", "imdb"]:
                save_logits = logits[:, :2]
                save_probs = probs[:, :2]

            if task_name == "NaturalLanguageInference" and ood_name in ["bio_nli", "doc_nli", "qnli"]:
                save_logits = None
                save_probs = torch.stack([probs[:, 0], probs[:, 1] + probs[:, 2]], dim=1)

            labels = inputs["label"]
            preds = torch.argmax(save_probs, dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(save_probs.cpu().tolist())

            if save_logits is not None:
                all_logits.extend(save_logits.cpu().tolist())

    prompt_model.train()

    acc = 100 * sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    print("acc on {}: {}".format(ood_name, acc), flush=True)

    probs_np = np.array(all_probs, dtype=np.float32)
    logits_np = np.array(all_logits, dtype=np.float32) if len(all_logits) > 0 else None

    if save_path is not None:
        _save_scores_csv(
            save_path=save_path,
            labels=all_labels,
            preds=all_preds,
            probs=probs_np,
            logits=logits_np,
        )

    return acc


def eval(prompt_model, processor, dataset_path, mytemplate, tokenizer, WrapperClass, result_path,
         task_name, ood_list, dataset_name, model_name, logits_dir=None):
    print("evaluation")

    dataset = {}
    for ood_name in ood_list:
        dataset[ood_name] = processor.get_examples(os.path.join(dataset_path, ood_name), "test")

    dataloader_dict = {}
    for ood_name in dataset.keys():
        if os.path.exists(f"./datasets/tokenize/{task_name}/{ood_name}.pt"):
            print(f"load tokenized test dataset of {ood_name}")
            test_dataloader = torch.load(
                f"./datasets/tokenize/{task_name}/{ood_name}.pt",
                weights_only=False
            )
        else:
            test_dataloader = PromptDataLoader(
                dataset=dataset[ood_name],
                template=mytemplate,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=256,
                decoder_max_length=3,
                batch_size=2,
                shuffle=False,
                teacher_forcing=False,
                predict_eos_token=False,
                truncate_method="tail"
            )
            batch_list = []
            for batch in test_dataloader:
                batch_list.append(batch)
            os.makedirs(f"./datasets/tokenize/{task_name}", exist_ok=True)
            print(f"save tokenized test dataset of {ood_name}")
            torch.save(batch_list, f"./datasets/tokenize/{task_name}/{ood_name}.pt")
            test_dataloader = batch_list

        dataloader_dict[ood_name] = test_dataloader

    print("Performance:")

    names = ["Dataset"]
    accuracies = ["Acc"]

    for ood_name, test_dataloader in dataloader_dict.items():
        save_path = None
        if logits_dir is not None:
            os.makedirs(logits_dir, exist_ok=True)
            save_path = os.path.join(logits_dir, f"{ood_name}_logits.csv")

        acc = evaluation(
            test_dataloader,
            prompt_model,
            task_name,
            dataset_name,
            model_name,
            ood_name,
            save_path=save_path
        )

        names.append(ood_name)
        accuracies.append(acc)

    results = pd.DataFrame([accuracies], columns=names)
    results.to_csv(result_path, sep="\t", index=False, header=names)
    print("finish evaluation", flush=True)
