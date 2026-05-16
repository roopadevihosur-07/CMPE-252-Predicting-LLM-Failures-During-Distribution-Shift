import re
import pandas as pd
from pathlib import Path

EXP1_FILE = "amazon_base_112241_exp1.out"
EXP2_FILE = "amazon_base_112291_exp2.out"

task_metric = {
    "Sentiment": "acc",
    "Toxic Detection": "acc",
    "Natural Language Inference": "acc",
    "Named Entity Recognition": "f1",
    "Question Answering": "f1",
}

source_to_task = {
    "amazon": "Sentiment",
    "civil_comments": "Toxic Detection",
    "mnli": "Natural Language Inference",
    "fewnerd": "Named Entity Recognition",
    "squad": "Question Answering",
}

exp2_targets = {
    "dynasent", "imdb", "semeval", "sst5", "yelp",
    "abuse_analyzer", "civil_comments", "implicit_hate", "toxigen",
    "anli", "contract_nli", "wanli",
    "conll", "ener", "wnut",
    "advqa", "searchqa"
}

def parse_exp1(path: str) -> pd.DataFrame:
    rows = []

    source_pat = re.compile(r"^(amazon|civil_comments|mnli|fewnerd|squad)$")
    metric_pat = re.compile(r"^(acc|f1|exact_match) on ([A-Za-z0-9_]+): ([0-9.]+)$")

    current_source = None
    current_task = None

    lines = Path(path).read_text(errors="ignore").splitlines()
    for line in lines:
        line = line.strip()

        m = source_pat.match(line)
        if m:
            current_source = m.group(1)
            current_task = source_to_task[current_source]
            continue

        m = metric_pat.match(line)
        if m and current_source and current_task:
            metric, target, score = m.group(1), m.group(2), float(m.group(3))
            if metric == task_metric[current_task]:
                rows.append({
                    "task": current_task,
                    "source": current_source,
                    "target": target,
                    "metric": metric,
                    "score": score
                })

    return pd.DataFrame(rows)

def parse_exp2(path: str) -> pd.DataFrame:
    rows = []

    task_pat = re.compile(r"^=+\s*(.*?)\s*=+$")
    metric_pat = re.compile(r"^(acc|f1|exact_match) on ([A-Za-z0-9_]+): ([0-9.]+)$")

    current_task = None
    current_target = None

    lines = Path(path).read_text(errors="ignore").splitlines()
    for line in lines:
        line = line.strip()

        m = task_pat.match(line)
        if m:
            current_task = m.group(1)
            current_target = None
            continue

        if line in exp2_targets:
            current_target = line
            continue

        m = metric_pat.match(line)
        if m and current_task and current_target:
            metric, eval_name, score = m.group(1), m.group(2), float(m.group(3))
            if metric == task_metric[current_task] and eval_name == current_target:
                rows.append({
                    "task": current_task,
                    "target": current_target,
                    "metric": metric,
                    "tt_score": score
                })

    return pd.DataFrame(rows)

def compute_td(exp1: pd.DataFrame, exp2: pd.DataFrame, tau: float = 5.0) -> pd.DataFrame:
    # Convert NER F1 from 0-1 to 0-100 so threshold is comparable
    exp1 = exp1.copy()
    exp2 = exp2.copy()

    exp1.loc[exp1["task"] == "Named Entity Recognition", "score"] *= 100.0
    exp2.loc[exp2["task"] == "Named Entity Recognition", "tt_score"] *= 100.0

    m_ss = exp1[exp1["source"] == exp1["target"]][["task", "source", "metric", "score"]].rename(
        columns={"source": "source_ds", "score": "M_ss"}
    )

    m_st = exp1[exp1["source"] != exp1["target"]][["task", "source", "target", "metric", "score"]].rename(
        columns={"source": "source_ds", "target": "target_ds", "score": "M_st"}
    )

    m_tt = exp2.rename(columns={"target": "target_ds", "tt_score": "M_tt"})

    df = m_st.merge(m_ss, on=["task", "metric", "source_ds"], how="left")
    df = df.merge(m_tt, on=["task", "metric", "target_ds"], how="inner")

    df["SD"] = df["M_ss"] - df["M_st"]
    df["TD"] = df["M_tt"] - df["M_st"]
    df["IDD"] = df["M_ss"] - df["M_tt"]

    def label(row):
        sd_big = row["SD"] >= tau
        td_big = row["TD"] >= tau
        if sd_big and td_big:
            return "Classic"
        elif sd_big and not td_big:
            return "Observed"
        elif not sd_big and td_big:
            return "Unobserved"
        else:
            return "No-Challenge"

    df["category"] = df.apply(label, axis=1)
    return df.sort_values(["task", "source_ds", "target_ds"]).reset_index(drop=True)

def main():
    exp1 = parse_exp1(EXP1_FILE)
    exp2 = parse_exp2(EXP2_FILE)

    print("\nParsed Exp 1:")
    print(exp1.to_string(index=False))

    print("\nParsed Exp 2:")
    print(exp2.to_string(index=False))

    td = compute_td(exp1, exp2, tau=5.0)

    td.to_csv("experiment2_td_analysis.csv", index=False)

    print("\nTD analysis:")
    print(td[["task", "source_ds", "target_ds", "metric", "M_ss", "M_st", "M_tt", "SD", "TD", "IDD", "category"]].to_string(index=False))

    print("\nCategory counts:")
    print(td["category"].value_counts())

    print("\nSaved: experiment2_td_analysis.csv")

if __name__ == "__main__":
    main()
