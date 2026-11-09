from src.graph_retriever import GraphResult, GraphRetriever
from src.types import SearchHit
from src.vector_store import VectorStore


def reciprocal_rank_fusion(
    rankings: list[tuple[list[SearchHit], float]], k: int, top_k: int
) -> list[SearchHit]:
    # Weighted RRF: score(d) = sum_r w_r / (k + rank_r(d)). Rank-based, so the
    # incomparable vector and graph score scales never need to be calibrated.
    scores: dict[str, float] = {}
    hits_by_id: dict[str, SearchHit] = {}
    for hits, weight in rankings:
        for rank, hit in enumerate(hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + weight / (k + rank)
            hits_by_id.setdefault(hit["id"], hit)
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [{**hits_by_id[doc_id], "score": scores[doc_id]} for doc_id in ranked]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_retriever: GraphRetriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def retrieve_batch(self, questions: list[str], top_k: int) -> list[GraphResult]:
        vector_ranked = self.vector_store.search_batch(questions, self.candidate_k)
        results = []
        for question, vector_hits in zip(questions, vector_ranked):
            graph = self.graph_retriever.retrieve(question, self.candidate_k)
            fused = reciprocal_rank_fusion(
                [(vector_hits, self.vector_weight), (graph.hits, self.graph_weight)],
                self.rrf_k,
                top_k,
            )
            results.append(GraphResult(fused, graph.facts, graph.seeds))
        return results
