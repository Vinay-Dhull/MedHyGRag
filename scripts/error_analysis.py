import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_pubmedqa_labeled  # noqa: E402
from src.evaluate import LABELS  # noqa: E402

CONFIGS = ("plain_llm", "vector_rag", "graph_rag", "hybrid_rrf", "hybrid_facts", "oracle_rag")
RETRIEVAL_CONFIGS = ("vector_rag", "graph_rag", "hybrid_rrf", "hybrid_facts")
BASELINE = "vector_rag"


def load_predictions(results_dir: Path, prompt: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in CONFIGS:
        path = results_dir / f"preds_{name}_{prompt}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        df = df.drop_duplicates("id").set_index("id")
        df["correct"] = df["label"] == df["prediction"]
        df["retrieved"] = df["retrieved_ids"].map(lambda ids: [i for i in ids.split(";") if i])
        df["gold_rank"] = [
            retrieved.index(f"labeled_{qid}") + 1 if f"labeled_{qid}" in retrieved else 0
            for qid, retrieved in zip(df.index, df["retrieved"])
        ]
        frames[name] = df

    if BASELINE not in frames:
        raise SystemExit(f"{BASELINE} predictions are required as the reference system")
    shared = sorted(set.intersection(*(set(df.index) for df in frames.values())))
    return {name: df.loc[shared] for name, df in frames.items()}


def per_label_metrics(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(LABELS)
    report, confusion = [], []
    for name, df in frames.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            df["label"], df["prediction"], labels=labels, zero_division=0
        )
        for i, label in enumerate(labels):
            report.append({
                "config": name, "label": label,
                "precision": round(precision[i], 4), "recall": round(recall[i], 4),
                "f1": round(f1[i], 4), "support": int(support[i]),
                "predicted_share": round((df["prediction"] == label).mean(), 4),
            })
        matrix = confusion_matrix(df["label"], df["prediction"], labels=labels)
        for g, gold in enumerate(labels):
            for p, predicted in enumerate(labels):
                confusion.append({
                    "config": name, "gold": gold, "predicted": predicted, "count": int(matrix[g, p]),
                })
    return pd.DataFrame(report), pd.DataFrame(confusion)


def discordance_vs_baseline(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = frames[BASELINE]
    rows = []
    for name, df in frames.items():
        if name == BASELINE:
            continue
        rows.append({
            "config": name,
            "both_correct": int((base["correct"] & df["correct"]).sum()),
            "only_vector_correct": int((base["correct"] & ~df["correct"]).sum()),
            "only_config_correct": int((~base["correct"] & df["correct"]).sum()),
            "both_wrong": int((~base["correct"] & ~df["correct"]).sum()),
            "prediction_changed": int((base["prediction"] != df["prediction"]).sum()),
        })
    return pd.DataFrame(rows)


def accuracy_by_gold_rank(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Separates "the right abstract was missing" from "it was there but the answer was wrong".
    buckets = {
        "gold_rank_1": lambda rank: rank == 1,
        "gold_rank_2_to_5": lambda rank: rank >= 2,
        "gold_missing": lambda rank: rank == 0,
    }
    rows = []
    for name in RETRIEVAL_CONFIGS:
        if name not in frames:
            continue
        df = frames[name]
        for bucket, rule in buckets.items():
            mask = rule(df["gold_rank"])
            rows.append({
                "config": name, "bucket": bucket, "n": int(mask.sum()),
                "accuracy": round(df.loc[mask, "correct"].mean(), 4) if mask.any() else None,
            })
    return pd.DataFrame(rows)


def vector_error_taxonomy(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Classifies each vector RAG error by whether retrieval or generation was at fault,
    # using the oracle (gold abstract only) as the counterfactual.
    if "oracle_rag" not in frames:
        return pd.DataFrame()
    vector, oracle = frames[BASELINE], frames["oracle_rag"]
    wrong = ~vector["correct"]
    retrieved = vector["gold_rank"] > 0
    categories = {
        "retrieval_miss_oracle_correct": wrong & ~retrieved & oracle["correct"],
        "retrieval_miss_oracle_wrong": wrong & ~retrieved & ~oracle["correct"],
        "distractor_interference": wrong & retrieved & oracle["correct"],
        "generation_error": wrong & retrieved & ~oracle["correct"],
    }
    n_errors = int(wrong.sum())
    return pd.DataFrame([
        {
            "category": category, "count": int(mask.sum()),
            "share_of_errors": round(mask.sum() / n_errors, 4) if n_errors else None,
        }
        for category, mask in categories.items()
    ])


def rrf_evidence_changes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "hybrid_rrf" not in frames:
        return pd.DataFrame()
    vector, rrf = frames[BASELINE], frames["hybrid_rrf"]
    changed = pd.Series(
        [set(v) != set(h) for v, h in zip(vector["retrieved"], rrf["retrieved"])], index=vector.index
    )
    rows = []
    for group, mask in (("evidence_changed", changed), ("evidence_unchanged", ~changed)):
        rows.append({
            "group": group, "n": int(mask.sum()),
            "vector_accuracy": round(vector.loc[mask, "correct"].mean(), 4) if mask.any() else None,
            "rrf_accuracy": round(rrf.loc[mask, "correct"].mean(), 4) if mask.any() else None,
            "gold_gained": int((mask & (vector["gold_rank"] == 0) & (rrf["gold_rank"] > 0)).sum()),
            "gold_lost": int((mask & (vector["gold_rank"] > 0) & (rrf["gold_rank"] == 0)).sum()),
        })
    return pd.DataFrame(rows)


def discordant_examples(frames: dict[str, pd.DataFrame], questions: dict[str, str]) -> pd.DataFrame:
    base = frames[BASELINE]
    rows = []
    for name in ("graph_rag", "hybrid_rrf", "hybrid_facts"):
        if name not in frames:
            continue
        df = frames[name]
        for qid in base.index[base["correct"] != df["correct"]]:
            rows.append({
                "config": name,
                "winner": name if df.at[qid, "correct"] else BASELINE,
                "question_id": qid,
                "question": questions.get(qid, ""),
                "gold": base.at[qid, "label"],
                "vector_prediction": base.at[qid, "prediction"],
                "config_prediction": df.at[qid, "prediction"],
                "vector_gold_rank": base.at[qid, "gold_rank"],
                "config_gold_rank": df.at[qid, "gold_rank"],
                "vector_rationale": base.at[qid, "rationale"],
                "config_rationale": df.at[qid, "rationale"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis of saved predictions (no GPU)")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--prompt", default="v3")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    results_dir = ROOT / config["paths"]["results_dir"] / args.split
    output_dir = results_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_predictions(results_dir, args.prompt)
    _, labeled = load_pubmedqa_labeled()
    questions = {q["id"]: q["question"] for q in labeled}
    print(f"Configs: {', '.join(frames)} | shared questions: {len(next(iter(frames.values())))}")

    report, confusion = per_label_metrics(frames)
    tables = {
        "vector_error_taxonomy": vector_error_taxonomy(frames),
        "accuracy_by_gold_rank": accuracy_by_gold_rank(frames),
        "discordance_vs_vector": discordance_vs_baseline(frames),
        "rrf_evidence_changes": rrf_evidence_changes(frames),
        "per_label_metrics": report,
    }
    for title, table in tables.items():
        table.to_csv(output_dir / f"{title}.csv", index=False)
        if not table.empty:
            print(f"\n=== {title} ===\n{table.to_string(index=False)}")

    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)
    examples = discordant_examples(frames, questions)
    examples.to_csv(output_dir / "discordant_examples.csv", index=False)
    print(f"\nSaved {len(examples)} discordant examples and all tables to {output_dir}")


if __name__ == "__main__":
    main()
