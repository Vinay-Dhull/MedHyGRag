import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import build_corpus, load_pubmedqa_labeled  # noqa: E402
from src.faithfulness import FaithfulnessScorer  # noqa: E402

AUDIT_CONFIGS = ("vector_rag", "hybrid_facts", "graph_rag")
ROWS_PER_CONFIG = 60
SHEET_NAME = "faithfulness_audit_sheet.csv"
KEY_NAME = "faithfulness_audit_key.csv"


def export(config: dict, results_dir: Path, prompt: str, per_label: int) -> None:
    seed = config["seed"]
    _, questions = load_pubmedqa_labeled()
    question_text = {q["id"]: q["question"] for q in questions}
    corpus = {
        doc["id"]: doc["text"]
        for doc in build_corpus(config["corpus"]["num_distractors"], seed)
    }
    scorer = FaithfulnessScorer()

    candidates = []
    for name in AUDIT_CONFIGS:
        preds = pd.read_csv(
            results_dir / f"preds_{name}_{prompt}.csv", dtype={"id": str}, keep_default_na=False
        ).drop_duplicates("id")
        sample = preds.sample(n=min(ROWS_PER_CONFIG, len(preds)), random_state=seed)
        contexts = [
            [corpus[i] for i in ids.split(";") if i in corpus] if ids else []
            for ids in sample["retrieved_ids"]
        ]
        verdicts = scorer.judge(sample["rationale"].tolist(), contexts)
        for (_, row), passages, claims in zip(sample.iterrows(), contexts, verdicts):
            for verdict in claims:
                candidates.append({
                    "config": name,
                    "question_id": row["id"],
                    "question": question_text[row["id"]],
                    "claim": verdict["claim"],
                    "evidence": passages[verdict["passage"]],
                    "nli_entailment": round(verdict["entailment"], 3),
                    "nli_supported": "yes" if verdict["supported"] else "no",
                })

    rng = random.Random(seed)
    picked = []
    for label in ("yes", "no"):
        pool = [c for c in candidates if c["nli_supported"] == label]
        picked += rng.sample(pool, min(per_label, len(pool)))
    rng.shuffle(picked)

    audit = pd.DataFrame(picked)
    audit.insert(0, "audit_id", range(1, len(audit) + 1))
    # The NLI verdict goes into a separate key file so the annotator labels blind.
    audit[["audit_id", "question", "claim", "evidence"]].assign(
        human_supported="", notes=""
    ).to_csv(results_dir / SHEET_NAME, index=False)
    audit[["audit_id", "config", "question_id", "nli_entailment", "nli_supported"]].to_csv(
        results_dir / KEY_NAME, index=False
    )
    print(f"Wrote {len(audit)} claims to {results_dir / SHEET_NAME} (key: {KEY_NAME})")


def score(results_dir: Path) -> None:
    sheet = pd.read_csv(results_dir / SHEET_NAME, dtype=str, keep_default_na=False)
    key = pd.read_csv(results_dir / KEY_NAME, dtype={"audit_id": str})
    # Spreadsheet round-trips can turn ids into text ("1", "1.0") and append blank rows.
    for frame in (sheet, key):
        frame["audit_id"] = frame["audit_id"].str.strip().str.replace(r"\.0$", "", regex=True)
    sheet = sheet[sheet["audit_id"] != ""]
    merged = key.merge(sheet[["audit_id", "human_supported"]], on="audit_id")
    merged["human_supported"] = merged["human_supported"].astype(str).str.strip().str.lower()
    labeled = merged[merged["human_supported"].isin(["yes", "no"])]
    if labeled.empty:
        raise SystemExit("No human labels found; fill the human_supported column with yes/no")

    agreement = (labeled["human_supported"] == labeled["nli_supported"]).mean()
    kappa = cohen_kappa_score(labeled["nli_supported"], labeled["human_supported"])
    print(f"Labeled claims: {len(labeled)} of {len(merged)}")
    print(f"Agreement: {agreement:.1%} | Cohen's kappa: {kappa:.3f}\n")
    print(pd.crosstab(labeled["nli_supported"], labeled["human_supported"],
                      rownames=["NLI"], colnames=["human"]))
    print("\nAgreement by configuration:")
    print(labeled.groupby("config").apply(
        lambda g: (g["human_supported"] == g["nli_supported"]).mean()
    ).round(3).to_string())

    pd.DataFrame([{
        "n": len(labeled), "agreement": round(agreement, 4), "cohen_kappa": round(kappa, 4),
    }]).to_csv(results_dir / "faithfulness_audit_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual audit of the NLI faithfulness judge")
    parser.add_argument("command", choices=("export", "score"))
    parser.add_argument("--prompt", default="v3")
    parser.add_argument("--per-label", type=int, default=20)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    results_dir = ROOT / config["paths"]["results_dir"] / "test"
    if args.command == "export":
        export(config, results_dir, args.prompt, args.per_label)
    else:
        score(results_dir)


if __name__ == "__main__":
    main()