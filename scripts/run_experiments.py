import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import abstract_text, load_pubmedqa_labeled, split_dev_test  # noqa: E402
from src.evaluate import LABELS, classification_scores  # noqa: E402
from src.llm import LocalLLM  # noqa: E402
from src.prompts import SYSTEM_PROMPTS, plain_messages, rag_messages  # noqa: E402
from src.types import Messages, Question  # noqa: E402

CONFIGS = ("plain_llm", "oracle_rag", "vector_rag", "graph_rag")
RETRIEVAL_CONFIGS = ("vector_rag", "graph_rag")
CHECKPOINT_SIZE = 50


@dataclass
class Resources:
    gold_texts: dict[str, str]
    vector_store: object | None = None
    graph_retriever: object | None = None


def load_resources(configs: list[str], config: dict) -> Resources:
    labeled, _ = load_pubmedqa_labeled()
    resources = Resources({str(row["pubid"]): abstract_text(row) for row in labeled})

    if "vector_rag" in configs:
        from src.vector_store import VectorStore

        resources.vector_store = VectorStore(ROOT / "config.yaml")
    if "graph_rag" in configs:
        from src.entities import EntityExtractor
        from src.graph_retriever import GraphRetriever
        from src.graph_store import GraphStore

        graph_cfg = config["graph"]
        resources.graph_retriever = GraphRetriever(
            GraphStore(),
            EntityExtractor(graph_cfg["spacy_model"]),
            graph_cfg["neighbours_per_entity"],
            graph_cfg["neighbour_weight"],
            graph_cfg["max_facts"],
        )
    return resources


def build_inputs(
    config_name: str,
    questions: list[Question],
    resources: Resources,
    prompt_version: str,
    top_k: int,
) -> tuple[list[Messages], list[list[str]]]:
    texts = [q["question"] for q in questions]
    match config_name:
        case "plain_llm":
            return [plain_messages(t, prompt_version) for t in texts], [[] for _ in texts]
        case "oracle_rag":
            messages = [
                rag_messages(q["question"], [resources.gold_texts[q["id"]]], prompt_version)
                for q in questions
            ]
            return messages, [[f"labeled_{q['id']}"] for q in questions]
        case "vector_rag":
            ranked = resources.vector_store.search_batch(texts, top_k)
            messages = [
                rag_messages(t, [hit["text"] for hit in hits], prompt_version)
                for t, hits in zip(texts, ranked)
            ]
            return messages, [[hit["id"] for hit in hits] for hits in ranked]
        case "graph_rag":
            results = [resources.graph_retriever.retrieve(t, top_k) for t in texts]
            # Questions with no graph match fall back to the plain prompt; coverage
            # is reported so this fallback is visible in the results.
            messages = [
                rag_messages(t, [hit["text"] for hit in r.hits], prompt_version, r.facts)
                if r.hits
                else plain_messages(t, prompt_version)
                for t, r in zip(texts, results)
            ]
            return messages, [[hit["id"] for hit in r.hits] for r in results]
    raise ValueError(f"Unknown config: {config_name}")


def run_config(
    config_name: str,
    llm: LocalLLM,
    resources: Resources,
    questions: list[Question],
    prompt_version: str,
    top_k: int,
    output_dir: Path,
) -> pd.DataFrame:
    path = output_dir / f"preds_{config_name}_{prompt_version}.csv"
    done_ids = set(pd.read_csv(path, dtype={"id": str})["id"]) if path.exists() else set()
    pending = [q for q in questions if q["id"] not in done_ids]

    for start in tqdm(range(0, len(pending), CHECKPOINT_SIZE), desc=config_name):
        chunk = pending[start:start + CHECKPOINT_SIZE]
        messages, retrieved = build_inputs(config_name, chunk, resources, prompt_version, top_k)
        pd.DataFrame(
            {
                "id": [q["id"] for q in chunk],
                "label": [q["label"] for q in chunk],
                "prediction": llm.classify(messages, list(LABELS)),
                "retrieved_ids": [";".join(ids) for ids in retrieved],
            }
        ).to_csv(path, mode="a", header=not path.exists(), index=False)

    predictions = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    return predictions[predictions["id"].isin({q["id"] for q in questions})]


def retrieval_stats(predictions: pd.DataFrame) -> dict[str, float]:
    retrieved = predictions["retrieved_ids"].str.split(";")
    gold = "labeled_" + predictions["id"]
    return {
        "coverage": round((predictions["retrieved_ids"] != "").mean(), 4),
        "gold_recall": round(
            sum(g in ids for g, ids in zip(gold, retrieved)) / len(predictions), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedHyGRag experiments on PubMedQA")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default=None)
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    prompt_version = args.prompt or config["prompt_version"]
    top_k = config["retrieval"]["top_k"]
    output_dir = ROOT / config["paths"]["results_dir"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    _, questions = load_pubmedqa_labeled()
    dev, test = split_dev_test(questions, config["splits"]["dev_size"], config["seed"])
    questions = dev if args.split == "dev" else test
    questions = questions[: args.limit] if args.limit else questions

    resources = load_resources(args.configs, config)
    llm = LocalLLM(ROOT / "config.yaml")

    labels = [q["label"] for q in questions]
    summary = [{"config": "majority_yes", **classification_scores(labels, ["yes"] * len(labels))}]
    for config_name in args.configs:
        preds = run_config(
            config_name, llm, resources, questions, prompt_version, top_k, output_dir
        )
        row = {
            "config": config_name,
            **classification_scores(preds["label"].tolist(), preds["prediction"].tolist()),
        }
        if config_name in RETRIEVAL_CONFIGS:
            row |= retrieval_stats(preds)
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    suffix = f"_limit{args.limit}" if args.limit else ""
    summary_df.to_csv(output_dir / f"summary_{prompt_version}{suffix}.csv", index=False)
    print(f"\n[{args.split} | prompt {prompt_version}]\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
