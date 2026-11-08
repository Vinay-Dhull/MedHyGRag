import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import build_corpus  # noqa: E402
from src.entities import EntityExtractor  # noqa: E402
from src.graph_store import (  # noqa: E402
    AURA_FREE_NODE_LIMIT,
    AURA_FREE_RELATIONSHIP_LIMIT,
    GraphStore,
)
from src.types import Document  # noqa: E402


def select_mentions(
    documents: list[Document],
    entity_sets: list[set[str]],
    min_df: int,
    max_df_ratio: float,
    max_per_doc: int,
) -> tuple[list[dict], list[dict]]:
    n_docs = len(documents)
    df = Counter(name for names in entity_sets for name in names)
    idf = {
        name: math.log(n_docs / count)
        for name, count in df.items()
        if count >= min_df and count / n_docs <= max_df_ratio
    }

    mentions = []
    for doc, names in zip(documents, entity_sets):
        # Keep each abstract's most specific entities; this bounds relationship
        # count for AuraDB Free and drops weakly informative terms.
        ranked = sorted((n for n in names if n in idf), key=lambda n: (-idf[n], n))
        mentions.extend({"doc_id": doc["id"], "entity": n} for n in ranked[:max_per_doc])

    used = sorted({m["entity"] for m in mentions})
    entities = [{"name": n, "idf": round(idf[n], 4), "df": df[n]} for n in used]
    return entities, mentions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MedHyGRag knowledge graph in Neo4j")
    parser.add_argument("--reset", action="store_true", help="delete the existing graph first")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    graph_cfg = config["graph"]

    documents = build_corpus(config["corpus"]["num_distractors"], config["seed"])
    extractor = EntityExtractor(graph_cfg["spacy_model"])
    entity_sets = extractor.extract([d["text"] for d in documents], show_progress=True)

    entities, mentions = select_mentions(
        documents,
        entity_sets,
        graph_cfg["min_df"],
        graph_cfg["max_df_ratio"],
        graph_cfg["max_entities_per_doc"],
    )
    n_nodes = len(documents) + len(entities)
    print(f"Nodes: {n_nodes:,} ({len(entities):,} entities) | Relationships: {len(mentions):,}")
    if n_nodes > AURA_FREE_NODE_LIMIT or len(mentions) > AURA_FREE_RELATIONSHIP_LIMIT:
        raise SystemExit("Exceeds AuraDB Free limits; lower graph.max_entities_per_doc")

    document_rows = [
        {"id": d["id"], "text": d["text"], "source": d["metadata"]["source"]} for d in documents
    ]
    with GraphStore() as store:
        if args.reset:
            store.reset()
        store.load(document_rows, entities, mentions)
    print("Graph loaded")


if __name__ == "__main__":
    main()
