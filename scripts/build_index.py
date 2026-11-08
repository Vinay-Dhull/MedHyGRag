import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import build_corpus, load_pubmedqa_labeled, split_dev_test  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

K_VALUES = (1, 3, 5, 10)
QUERY_BATCH_SIZE = 64


def recall_at_k(store: VectorStore, questions: list[dict[str, str]]) -> pd.DataFrame:
    hits = dict.fromkeys(K_VALUES, 0)
    for start in tqdm(range(0, len(questions), QUERY_BATCH_SIZE), desc="Retrieval eval"):
        batch = questions[start:start + QUERY_BATCH_SIZE]
        results = store.search_batch([q["question"] for q in batch], top_k=max(K_VALUES))
        for question, ranked in zip(batch, results):
            ranked_ids = [hit["id"] for hit in ranked]
            gold_id = f"labeled_{question['id']}"
            for k in K_VALUES:
                hits[k] += gold_id in ranked_ids[:k]

    return pd.DataFrame(
        {"k": list(K_VALUES), "recall": [round(hits[k] / len(questions), 4) for k in K_VALUES]}
    )


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())

    store = VectorStore(ROOT / "config.yaml")
    store.build(build_corpus(config["corpus"]["num_distractors"], config["seed"]))

    _, questions = load_pubmedqa_labeled()
    _, test = split_dev_test(questions, config["splits"]["dev_size"], config["seed"])

    recall = recall_at_k(store, test)
    output_dir = ROOT / config["paths"]["results_dir"]
    output_dir.mkdir(exist_ok=True)
    recall.to_csv(output_dir / "phase1_vector_recall.csv", index=False)

    print(f"\nVector Recall@k on {len(test)} test questions\n{recall.to_string(index=False)}")


if __name__ == "__main__":
    main()
