import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import Question, load_pubmedqa_labeled, split_dev_test  # noqa: E402
from src.evaluate import LABELS, classification_scores  # noqa: E402
from src.llm import LocalLLM  # noqa: E402
from src.prompts import Messages, plain_messages, rag_messages  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

CONFIGS = ("plain_llm", "vector_rag")
CHECKPOINT_SIZE = 50


def build_inputs(
    config_name: str, questions: list[Question], store: VectorStore | None, top_k: int
) -> tuple[list[Messages], list[list[str]]]:
    if config_name == "plain_llm":
        return [plain_messages(q["question"]) for q in questions], [[] for _ in questions]

    hits = store.search_batch([q["question"] for q in questions], top_k)
    messages = [
        rag_messages(q["question"], [hit["text"] for hit in ranked])
        for q, ranked in zip(questions, hits)
    ]
    return messages, [[hit["id"] for hit in ranked] for ranked in hits]


def run_config(
    config_name: str,
    llm: LocalLLM,
    store: VectorStore | None,
    questions: list[Question],
    top_k: int,
    output_dir: Path,
) -> pd.DataFrame:
    # Predictions are appended per checkpoint so an interrupted Kaggle session resumes
    # where it stopped instead of recomputing everything.
    path = output_dir / f"preds_{config_name}.csv"
    done_ids = set(pd.read_csv(path, dtype={"id": str})["id"]) if path.exists() else set()
    pending = [q for q in questions if q["id"] not in done_ids]

    for start in tqdm(range(0, len(pending), CHECKPOINT_SIZE), desc=config_name):
        chunk = pending[start:start + CHECKPOINT_SIZE]
        messages, retrieved = build_inputs(config_name, chunk, store, top_k)
        predictions = llm.classify(messages, list(LABELS))
        pd.DataFrame(
            {
                "id": [q["id"] for q in chunk],
                "label": [q["label"] for q in chunk],
                "prediction": predictions,
                "retrieved_ids": [";".join(ids) for ids in retrieved],
            }
        ).to_csv(path, mode="a", header=not path.exists(), index=False)

    predictions = pd.read_csv(path, dtype={"id": str})
    wanted = {q["id"] for q in questions}
    return predictions[predictions["id"].isin(wanted)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain LLM vs vector RAG on PubMedQA")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--limit", type=int, help="evaluate only the first N test questions")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    output_dir = ROOT / config["paths"]["results_dir"]
    output_dir.mkdir(exist_ok=True)

    _, questions = load_pubmedqa_labeled()
    _, test = split_dev_test(questions, config["splits"]["dev_size"], config["seed"])
    test = test[: args.limit] if args.limit else test

    store = VectorStore(ROOT / "config.yaml") if "vector_rag" in args.configs else None
    llm = LocalLLM(ROOT / "config.yaml")

    summary = []
    for config_name in args.configs:
        preds = run_config(
            config_name, llm, store, test, config["retrieval"]["top_k"], output_dir
        )
        scores = classification_scores(preds["label"].tolist(), preds["prediction"].tolist())
        summary.append({"config": config_name, **scores})

    summary_df = pd.DataFrame(summary)
    suffix = f"_limit{args.limit}" if args.limit else ""
    summary_df.to_csv(output_dir / f"phase1_baselines{suffix}.csv", index=False)
    print(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
