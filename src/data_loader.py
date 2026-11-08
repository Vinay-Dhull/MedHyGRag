import random
from typing import Any

from datasets import load_dataset

from src.types import Document, Question

DATASET = "qiaojin/PubMedQA"


def load_pubmedqa_labeled() -> tuple[Any, list[Question]]:
    ds = load_dataset(DATASET, "pqa_labeled", split="train")
    questions = [
        {"id": str(row["pubid"]), "question": row["question"], "label": row["final_decision"]}
        for row in ds
    ]
    return ds, questions


def split_dev_test(
    questions: list[Question], dev_size: int = 100, seed: int = 42
) -> tuple[list[Question], list[Question]]:
    shuffled = questions[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:dev_size], shuffled[dev_size:]


def abstract_text(row: dict[str, Any]) -> str:
    # Only the abstract sections are used. `long_answer` holds the study's
    # conclusion and would leak the gold label.
    return " ".join(row["context"]["contexts"])


def _to_document(row: dict[str, Any], source: str) -> Document:
    return {
        "id": f"{source}_{row['pubid']}",
        "text": abstract_text(row),
        "metadata": {"pubid": str(row["pubid"]), "source": source},
    }


def build_corpus(num_distractors: int = 20_000, seed: int = 42) -> list[Document]:
    labeled, _ = load_pubmedqa_labeled()
    documents = [_to_document(row, "labeled") for row in labeled]
    labeled_ids = {str(row["pubid"]) for row in labeled}

    unlabeled = load_dataset(DATASET, "pqa_unlabeled", split="train")
    order = list(range(len(unlabeled)))
    random.Random(seed).shuffle(order)

    distractors = []
    for idx in order:
        if len(distractors) == num_distractors:
            break
        row = unlabeled[idx]
        if str(row["pubid"]) not in labeled_ids:
            distractors.append(_to_document(row, "unlabeled"))

    print(f"Corpus: {len(documents)} gold + {len(distractors)} distractors")
    return documents + distractors
