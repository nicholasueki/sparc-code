"""Semantic memory mirror on Node C: sqlite-vec + fastembed.

Canonical facts live in Node A's SQLite (design §7.1). This store holds the
embedding index + statements for retrieval, keyed by Node A's fact_id, and is
rebuildable at any time from Node A.
"""
from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path

log = logging.getLogger("sparc.memory")


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SemanticMemory:
    def __init__(self, db_path: str, embed_model: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        import sqlite_vec
        from fastembed import TextEmbedding

        self.embedder = TextEmbedding(model_name=embed_model)
        self.dim = len(self.embed_one("dimension probe"))

        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS fact_text(
                fact_id TEXT PRIMARY KEY, statement TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS fact_vec USING vec0(
                fact_id TEXT PRIMARY KEY, embedding float[{self.dim}]);
            """
        )
        log.info("semantic memory ready (%s, dim=%d)", db_path, self.dim)

    def embed_one(self, text: str) -> list[float]:
        return list(next(iter(self.embedder.embed([text]))))

    def upsert(self, fact_id: str, statement: str) -> None:
        vec = self.embed_one(statement)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO fact_text(fact_id, statement) VALUES(?,?)",
                (fact_id, statement),
            )
            self.db.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fact_id,))
            self.db.execute(
                "INSERT INTO fact_vec(fact_id, embedding) VALUES(?,?)",
                (fact_id, _serialize_f32(vec)),
            )

    def delete(self, fact_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM fact_text WHERE fact_id=?", (fact_id,))
            self.db.execute("DELETE FROM fact_vec WHERE fact_id=?", (fact_id,))

    def retrieve(self, situation: str, k: int, min_score: float) -> list[tuple[str, str, float]]:
        """-> [(fact_id, statement, score)] score = 1 - cosine_distance."""
        vec = self.embed_one(situation)
        rows = self.db.execute(
            """
            SELECT v.fact_id, t.statement, v.distance
            FROM fact_vec v JOIN fact_text t USING(fact_id)
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_f32(vec), k),
        ).fetchall()
        out = []
        for fact_id, statement, dist in rows:
            score = 1.0 - float(dist) / 2.0  # vec0 cosine distance in [0,2]
            if score >= min_score:
                out.append((fact_id, statement, round(score, 3)))
        return out

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM fact_text").fetchone()[0]
