import re
from typing import Dict, List, Optional, Tuple


LABEL_ID_TO_NAME = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

LABEL_NAME_TO_ID = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}


def normalize_label(label: str) -> Optional[str]:
    if label is None:
        return None

    x = label.strip().lower()
    x = re.sub(r"[^a-z_ -]", "", x)
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"\s+", " ", x).strip()

    aliases = {
        "negative": "negative",
        "neg": "negative",
        "positive": "positive",
        "pos": "positive",
        "neutral": "neutral",
        "neu": "neutral",
    }

    return aliases.get(x)


def build_sentiment_prompt(shots: List[Dict], text: str) -> str:
    if len(shots) != 4:
        raise ValueError(f"Expected exactly 4 shots, got {len(shots)}")

    sections = []
    sections.append("sentiment classification")
    sections.append("Possible labels: negative, neutral, positive.")

    for shot in shots:
        label_name = LABEL_ID_TO_NAME[int(shot["label"])]
        sections.append(
            f"Text: {shot['text']}\n"
            f"Label: {label_name}"
        )

    sections.append(
        f"Text: {text}\n"
        "Answer exactly in this format:\n"
        "Label: <negative|neutral|positive>\n"
        "Confidence: <0-100>"
    )

    return "\n\n".join(sections)


def parse_llm_response(raw_text: str) -> Tuple[Optional[str], Optional[float]]:
    if raw_text is None:
        return None, None

    text = raw_text.strip()

    label = None
    confidence = None

    label_match = re.search(r"Label\s*:\s*([A-Za-z _-]+)", text, flags=re.IGNORECASE)
    if label_match:
        label = normalize_label(label_match.group(1))

    if label is None:
        for candidate in ["negative", "neutral", "positive"]:
            if re.search(rf"\b{candidate}\b", text, flags=re.IGNORECASE):
                label = candidate
                break

    conf_match = re.search(r"Confidence\s*:\s*([0-9]{1,3})", text, flags=re.IGNORECASE)
    if conf_match:
        confidence = float(conf_match.group(1))
    else:
        percent_match = re.search(r"\b([0-9]{1,3})\s*%\b", text)
        if percent_match:
            confidence = float(percent_match.group(1))
        else:
            any_num = re.search(r"\b([0-9]{1,3})\b", text)
            if any_num:
                confidence = float(any_num.group(1))

    if confidence is not None:
        confidence = max(0.0, min(100.0, confidence))

    return label, confidence
