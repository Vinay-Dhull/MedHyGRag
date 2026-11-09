import os
from collections.abc import Iterator
from typing import Any

from neo4j import GraphDatabase, RoutingControl
from tqdm import tqdm

AURA_FREE_NODE_LIMIT = 200_000
AURA_FREE_RELATIONSHIP_LIMIT = 400_000

_SCHEMA = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
)
_LOAD_DOCUMENTS = """
UNWIND $rows AS row
MERGE (d:Document {id: row.id})
SET d.text = row.text, d.source = row.source
"""
_LOAD_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {name: row.name})
SET e.idf = row.idf, e.df = row.df
"""
_LOAD_MENTIONS = """
UNWIND $rows AS row
MATCH (d:Document {id: row.doc_id})
MATCH (e:Entity {name: row.entity})
MERGE (d)-[:MENTIONS]->(e)
"""


def _batched(rows: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class GraphStore:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
            notifications_min_severity="OFF",
        )
        self.driver.verify_connectivity()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.driver.close()

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher, parameters_=params, routing_=RoutingControl.READ
        )
        return [record.data() for record in records]

    def count_nodes(self, label: str) -> int:
        rows = self.query(f"MATCH (n:{label}) RETURN count(n) AS n")
        return rows[0]["n"]

    def reset(self) -> None:
        with self.driver.session() as session:
            session.run(
                "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
            ).consume()

    def load(
        self,
        documents: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        mentions: list[dict[str, str]],
    ) -> None:
        for statement in _SCHEMA:
            self.driver.execute_query(statement)
        self._write(_LOAD_DOCUMENTS, documents, 1_000, "Documents")
        self._write(_LOAD_ENTITIES, entities, 5_000, "Entities")
        self._write(_LOAD_MENTIONS, mentions, 5_000, "Mentions")

    def _write(self, cypher: str, rows: list[Any], batch_size: int, label: str) -> None:
        batches = list(_batched(rows, batch_size))
        for batch in tqdm(batches, desc=label):
            self.driver.execute_query(cypher, parameters_={"rows": batch})