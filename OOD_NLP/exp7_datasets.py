from typing import Dict, List
import pandas as pd


def load_sentiment_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    expected = {"Text", "Label"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} missing required columns. Found: {df.columns.tolist()}")

    df = df[["Text", "Label"]].copy()
    df = df.rename(columns={"Text": "text", "Label": "label"})
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def sample_4_shots_balanced(df: pd.DataFrame, seed: int = 0) -> List[Dict]:
    # Aim for one per class + one extra from a random class.
    shots: List[Dict] = []

    for label in [0, 1, 2]:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            raise ValueError(f"No examples for label {label} in source dataset.")
        shot = sub.sample(n=1, random_state=seed + label).iloc[0]
        shots.append({"text": shot["text"], "label": int(shot["label"])})

    extra = df.sample(n=1, random_state=seed + 99).iloc[0]
    shots.append({"text": extra["text"], "label": int(extra["label"])})

    return shots


def build_eval_records(
    df: pd.DataFrame,
    source_name: str,
    target_name: str,
    shift_type: str,
    limit: int = 100,
) -> List[Dict]:
    if limit > 0:
        df = df.iloc[:limit].copy()

    records: List[Dict] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        records.append(
            {
                "example_id": idx,
                "source_dataset": source_name,
                "target_dataset": target_name,
                "shift_type": shift_type,
                "text": row["text"],
                "gold_label": int(row["label"]),
            }
        )
    return records
