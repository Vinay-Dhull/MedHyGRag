import re
from dataclasses import dataclass, field

from src.entities import EntityExtractor, normalize
from src.graph_store import GraphStore
from src.types import SearchHit

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'/][a-z0-9]+)*")
MAX_NGRAM = 4

_KNOWN_ENTITIES = "MATCH (e:Entity) WHERE e.name IN $names RETURN e.name AS name"

# One-hop expansion through shared documents. Neighbours are ranked by
# co-mention count weighted by IDF, so frequent but generic terms don't dominate.
_NEIGHBOURS = """
UNWIND $names AS name
MATCH (:Entity {name: name})<-[:MENTIONS]-(:Document)-[:MENTIONS]->(n:Entity)
WHERE NOT n.name IN $names
WITH name, n, count(*) AS shared
WHERE shared >= 2
ORDER BY shared * n.idf DESC
WITH name, collect({neighbour: n.name, shared: shared})[..$per_entity] AS top
UNWIND top AS row
RETURN name AS source, row.neighbour AS neighbour, row.shared AS shared
ORDER BY shared DESC
"""

# With $normalize, scores are divided by sqrt(entity count) so abstracts that mention
# many entities don't win simply by matching more terms by chance.
_RANK_DOCUMENTS = """
UNWIND $seeds AS seed
MATCH (e:Entity {name: seed.name})<-[:MENTIONS]-(d:Document)
WITH d, sum(e.idf * seed.weight) AS raw, collect(e.name) AS matched
WITH d, matched,
     CASE WHEN $normalize THEN raw / sqrt(toFloat(COUNT { (d)-[:MENTIONS]->() })) ELSE raw END AS score
ORDER BY score DESC, d.id
LIMIT $top_k
RETURN d.id AS id, d.text AS text, d.source AS source, score, matched
"""


@dataclass
class GraphResult:
    hits: list[SearchHit] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)


class GraphRetriever:
    def __init__(
        self,
        store: GraphStore,
        extractor: EntityExtractor,
        neighbours_per_entity: int = 3,
        neighbour_weight: float = 0.5,
        max_facts: int = 6,
        use_ngram_seeds: bool = True,
        normalize: bool = False,
    ) -> None:
        n_entities = store.count_nodes("Entity")
        if n_entities == 0:
            raise RuntimeError("Knowledge graph is empty; run scripts/build_graph.py first")
        print(f"Knowledge graph: {n_entities:,} entities")
        self.store = store
        self.extractor = extractor
        self.neighbours_per_entity = neighbours_per_entity
        self.neighbour_weight = neighbour_weight
        self.max_facts = max_facts
        self.use_ngram_seeds = use_ngram_seeds
        self.normalize = normalize

    def _seed_entities(self, question: str) -> list[str]:
        # NER alone misses many entities in short questions, so every question
        # n-gram is also matched against the graph's entity vocabulary.
        candidates = self.extractor.extract([question])[0]
        if self.use_ngram_seeds:
            tokens = _TOKEN.findall(question.lower())
            ngrams = {
                " ".join(tokens[i:i + n])
                for n in range(1, MAX_NGRAM + 1)
                for i in range(len(tokens) - n + 1)
            }
            candidates |= {name for c in ngrams if (name := normalize(c))}
        rows = self.store.query(_KNOWN_ENTITIES, names=sorted(candidates))
        return sorted(row["name"] for row in rows)

    def retrieve(self, question: str, top_k: int) -> GraphResult:
        seeds = self._seed_entities(question)
        if not seeds:
            return GraphResult()

        neighbours = self.store.query(
            _NEIGHBOURS, names=seeds, per_entity=self.neighbours_per_entity
        )
        weights = dict.fromkeys(seeds, 1.0)
        if self.neighbour_weight > 0:
            for row in neighbours:
                weights.setdefault(row["neighbour"], self.neighbour_weight)

        documents = self.store.query(
            _RANK_DOCUMENTS,
            seeds=[{"name": name, "weight": w} for name, w in weights.items()],
            top_k=top_k,
            normalize=self.normalize,
        )
        hits = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": {"source": doc["source"], "matched": doc["matched"]},
                "score": doc["score"],
            }
            for doc in documents
        ]
        facts = [
            f"{row['source']} is co-mentioned with {row['neighbour']} in {row['shared']} abstracts"
            for row in neighbours[: self.max_facts]
        ]
        return GraphResult(hits, facts, seeds)

    def search_batch(self, queries: list[str], top_k: int) -> list[list[SearchHit]]:
        return [self.retrieve(query, top_k).hits for query in queries]