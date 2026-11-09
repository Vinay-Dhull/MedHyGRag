import itertools
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

NEIGHBOUR_WEIGHTS = (0.0, 0.25, 0.5)
NGRAM_SEEDS = (True, False)
NORMALIZE = (False, True)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    graph_cfg = config["graph"]
    top_k = config["retrieval"]["top_k"]

    _, questions = load_pubmedqa_labeled()
    dev, _ = split_dev_test(questions, config["splits"]["dev_size"], config["seed"])

    store = GraphStore()
    extractor = EntityExtractor(graph_cfg["ner_model"], graph_cfg["ner_batch_size"])

    rows = []
    grid = list(itertools.product(NEIGHBOUR_WEIGHTS, NGRAM_SEEDS, NORMALIZE))
    for weight, ngrams, normalize in tqdm(grid, desc="Settings"):
        retriever = GraphRetriever(
            store, extractor, graph_cfg["neighbours_per_entity"], weight,
            graph_cfg["max_facts"], ngrams, normalize,
        )
        hits = covered = 0
        for q in dev:
            ids = [hit["id"] for hit in retriever.retrieve(q["question"], top_k).hits]
            covered += bool(ids)
            hits += f"labeled_{q['id']}" in ids
        rows.append({
            "neighbour_weight": weight,
            "ngram_seeds": ngrams,
            "normalize": normalize,
            "coverage": covered / len(dev),
            f"recall@{top_k}": hits / len(dev),
        })

    results = pd.DataFrame(rows).sort_values(f"recall@{top_k}", ascending=False)
    output_dir = ROOT / config["paths"]["results_dir"] / "dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "graph_retrieval_tuning.csv", index=False)
    print(f"\n{results.to_string(index=False)}")


if __name__ == "__main__":
    main()
