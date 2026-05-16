import os
import json
from tqdm import tqdm

dataset_name = "NewsQA"

for split in ["train", "test"]:
    raw_path = f"./datasets/raw/QuestionAnswering/{split}/{dataset_name}.jsonl"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    with open(raw_path, "rb") as f:
        examples = []
        for i, line in enumerate(f):
            if i == 0:
                continue
            examples.append(json.loads(line))

    out_name = dataset_name.lower()
    os.makedirs(f"./datasets/process/QuestionAnswering/{out_name}", exist_ok=True)

    with open(f"./datasets/process/QuestionAnswering/{out_name}/{split}.json", "w", newline="\n") as f:
        for example in tqdm(examples, desc=f"{split}-{dataset_name}"):
            title = ""
            context = example["context"]

            for qas in example["qas"]:
                ex_id = qas["id"]
                question = qas["question"]

                detected_texts = [da["text"] for da in qas.get("detected_answers", [])]
                all_answers = qas.get("answers", []) + detected_texts
                answers_text_list = list(set(all_answers))

                json.dump(
                    {
                        "title": title,
                        "context": context,
                        "id": ex_id,
                        "question": question,
                        "answers": {
                            "text": answers_text_list,
                            "answer_start": detected_texts,
                        },
                    },
                    f,
                )
                f.write("\n")

print("Finished rebuilding NewsQA.")
