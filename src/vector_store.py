from pathlib import Path
from typing import Any

import chromadb
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data_loader import Document

SearchHit = dict[str, Any]


class VectorStore:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        emb_cfg = self.config["embedding"]
        device = emb_cfg["device"] if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(emb_cfg["model_name"], device=device)
        self.batch_size = emb_cfg["batch_size"]
        self.default_top_k = self.config["retrieval"]["top_k"]

        client = chromadb.PersistentClient(path=self.config["paths"]["chroma_dir"])
        self.collection = client.get_or_create_collection(
            name=self.config["retrieval"]["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def build(self, documents: list[Document], chunk_size: int = 1_000) -> None:
        if self.collection.count() >= len(documents):
            print(f"Index already complete ({self.collection.count()} docs)")
            return

        for start in tqdm(range(0, len(documents), chunk_size), desc="Indexing"):
            chunk = documents[start:start + chunk_size]
            texts = [doc["text"] for doc in chunk]
            self.collection.upsert(
                ids=[doc["id"] for doc in chunk],
                documents=texts,
                metadatas=[doc["metadata"] for doc in chunk],
                embeddings=self.embed(texts),
            )

    def search_batch(self, queries: list[str], top_k: int | None = None) -> list[list[SearchHit]]:
        result = self.collection.query(
            query_embeddings=self.embed(queries),
            n_results=top_k or self.default_top_k,
        )
        return [
            [
                # Chroma returns cosine distance; convert to similarity for fusion later.
                {"id": doc_id, "text": text, "metadata": meta, "score": 1.0 - dist}
                for doc_id, text, meta, dist in zip(ids, texts, metas, dists)
            ]
            for ids, texts, metas, dists in zip(
                result["ids"], result["documents"], result["metadatas"], result["distances"]
            )
        ]

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        return self.search_batch([query], top_k)[0]