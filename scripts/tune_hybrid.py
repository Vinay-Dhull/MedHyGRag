import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_pubmedqa_labeled, split_dev_test  # noqa: E402
from src.entities import EntityExtractor  # noqa: E402
from src.graph_retriever import GraphRetriever  # noqa: E402
from src.graph_store import GraphStore  # noqa: E402
from src.hybrid import reciprocal_rank_fusion  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

GRAPH_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    graph_cfg, hybrid_cfg = config["graph"], config["hybrid"]
    top_k, candidate_k = config["retrieval"]["top_k"], hybrid_cfg["candidate_k"]

    _, questions = load_pubmedqa_labeled()
    dev, _ = split_dev_test(questions, config["splits"]["dev_size"], config["seed"])
    texts = [q["question"] for q in dev]
    gold = [f"labeled_{q['id']}" for q in dev]

    vector_ranked = VectorStore(ROOT / "config.yaml").search_batch(texts, candidate_k)
    graph = GraphRetriever(
        GraphStore(),
        EntityExtractor(graph_cfg["ner_model"], graph_cfg["ner_batch_size"]),
        graph_cfg["neighbours_per_entity"],
        graph_cfg["neighbour_weight"],
        graph_cfg["max_facts"],
        graph_cfg["use_ngram_seeds"],
        graph_cfg["normalize"],
    )
    # Candidates are fetched once; fusion for every weight is then pure arithmetic.
    graph_ranked = [graph.retrieve(t, candidate_k).hits for t in tqdm(texts, desc="Graph")]

    vector_hit = [g in {h["id"] for h in hits[:top_k]} for g, hits in zip(gold, vector_ranked)]
    rows = []
    for weight in GRAPH_WEIGHTS:
        hybrid_hit = [
            g in {h["id"] for h in reciprocal_rank_fusion(
                [(v, hybrid_cfg["vector_weight"]), (gr, weight)], hybrid_cfg["rrf_k"], top_k
            )}
            for g, v, gr in zip(gold, vector_ranked, graph_ranked)
        ]
        rows.append({
            "graph_weight": weight,
            f"recall@{top_k}": sum(hybrid_hit) / len(dev),
            "rescued": sum(h and not v for h, v in zip(hybrid_hit, vector_hit)),
            "lost": sum(v and not h for h, v in zip(hybrid_hit, vector_hit)),
        })

    results = pd.DataFrame(rows)
    output_dir = ROOT / config["paths"]["results_dir"] / "dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "hybrid_tuning.csv", index=False)
    print(f"\nVector-only recall@{top_k}: {sum(vector_hit) / len(dev):.2f}\n{results.to_string(index=False)}")


if __name__ == "__main__":
    main()
