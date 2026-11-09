import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import build_corpus  # noqa: E402
from src.evaluate import classification_scores  # noqa: E402
from src.faithfulness import FaithfulnessScorer  # noqa: E402

CONFIG_ORDER = (
    "plain_llm", "vector_rag", "graph_rag", "hybrid_rrf", "hybrid_facts", "oracle_rag",
)
BASELINE = "vector_rag"


def mcnemar_p(base_correct: pd.Series, other_correct: pd.Series) -> float:
    # Exact McNemar test on discordant pairs: is one system right where the other
    # is wrong more often than chance (p = 0.5) would explain?
    only_base = int((base_correct & ~other_correct).sum())
    only_other = int((~base_correct & other_correct).sum())
    if only_base + only_other == 0:
        return 1.0
    return binomtest(min(only_base, only_other), only_base + only_other, 0.5).pvalue


def load_predictions(results_dir: Path, prompt: str) -> dict[str, pd.DataFrame]:
    predictions = {}
    for name in CONFIG_ORDER:
        path = results_dir / f"preds_{name}_{prompt}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
            predictions[name] = df.drop_duplicates("id").set_index("id")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Final metrics, significance and faithfulness")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--prompt", default="v3")
    parser.add_argument("--faithfulness-sample", type=int, default=200)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    results_dir = ROOT / config["paths"]["results_dir"] / args.split
    predictions = load_predictions(results_dir, args.prompt)
    if not predictions:
        raise SystemExit(f"No v3 prediction files in {results_dir}")

    shared_ids = sorted(set.intersection(*(set(df.index) for df in predictions.values())))
    print(f"Configs: {', '.join(predictions)} | shared questions: {len(shared_ids)}")
    sample_ids = sorted(
        random.Random(config["seed"]).sample(shared_ids, min(args.faithfulness_sample, len(shared_ids)))
    )

    corpus = {doc["id"]: doc["text"] for doc in build_corpus(
        config["corpus"]["num_distractors"], config["seed"]
    )}
    scorer = FaithfulnessScorer()

    base = predictions.get(BASELINE)
    rows = []
    for name, df in predictions.items():
        df = df.loc[shared_ids]
        correct = df["label"] == df["prediction"]
        row = {"config": name, **classification_scores(df["label"].tolist(), df["prediction"].tolist())}
        if base is not None and name != BASELINE:
            base_correct = base.loc[shared_ids, "label"] == base.loc[shared_ids, "prediction"]
            row["mcnemar_p_vs_vector"] = round(mcnemar_p(base_correct, correct), 4)

        sample = df.loc[sample_ids]
        contexts = [
            [corpus[i] for i in ids.split(";") if i in corpus] if ids else []
            for ids in sample["retrieved_ids"]
        ]
        if any(contexts):
            scores = [s for s in scorer.score(sample["rationale"].tolist(), contexts) if s is not None]
            row["faithfulness"] = round(sum(scores) / len(scores), 4) if scores else None
            row["faithfulness_n"] = len(scores)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / f"evaluation_{args.prompt}.csv", index=False)
    print(f"\n[{args.split} | prompt {args.prompt}]\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
