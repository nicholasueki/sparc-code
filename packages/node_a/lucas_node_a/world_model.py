"""World model: canonical state. SQLite (durable) + hot dict (live).

M2 is the only writer (design LIM-M2-1). No ML here — identity matching is
cosine arithmetic; everything else is bookkeeping.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lucas_common.types import new_id

log = logging.getLogger("lucas.world")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT,
  track_id TEXT, present INTEGER DEFAULT 0, attending REAL DEFAULT 0,
  first_seen REAL, last_seen REAL, confidence REAL DEFAULT 0.5);
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY, ts REAL NOT NULL, type TEXT NOT NULL,
  description TEXT NOT NULL, entity_ids TEXT DEFAULT '[]', salience REAL DEFAULT 0.3);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
  description, content='events', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
  INSERT INTO events_fts(rowid, description) VALUES (new.rowid, new.description);
END;
CREATE TABLE IF NOT EXISTS facts(
  id TEXT PRIMARY KEY, statement TEXT NOT NULL, confidence REAL DEFAULT 0.6,
  source TEXT DEFAULT 'observed', created REAL, invalidated REAL);
CREATE TABLE IF NOT EXISTS known_faces(
  entity_id TEXT PRIMARY KEY, name TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS trace(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, deliberation_id TEXT,
  kind TEXT, payload TEXT);
"""


class WorldModel:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()
        # hot state, rebuilt from SQLite at boot
        self.present: dict[str, dict] = {}  # entity_id -> {name, track_id, since, attending}
        self.conversation: list[dict] = []  # [{role, text, ts}]
        self.last_action_ts: dict[str, float] = {}  # cooldown ledger
        self._rebuild_hot()

    def _rebuild_hot(self) -> None:
        for eid, name, track_id, last_seen in self.db.execute(
            "SELECT id, name, track_id, last_seen FROM entities WHERE present=1"
        ):
            # anything 'present' at boot is stale — mark absent, real presence re-derives
            self.db.execute("UPDATE entities SET present=0 WHERE id=?", (eid,))
        self.db.commit()

    # ---------------------------------------------------------- presence

    def person_appeared(self, track_id: str, embedding: Optional[list[float]] = None
                        ) -> tuple[str, Optional[str], str]:
        """-> (entity_id, name|None, identity: known|unknown|uncertain)"""
        name, identity, eid = None, "unknown", None
        if embedding is not None:
            match = self.match_face(embedding)
            if match:
                eid, name, identity = match
        if eid is None:
            eid = new_id()
            self.db.execute(
                "INSERT INTO entities(id, kind, track_id, present, first_seen, last_seen)"
                " VALUES(?, 'person', ?, 1, ?, ?)",
                (eid, track_id, time.time(), time.time()),
            )
        else:
            self.db.execute(
                "UPDATE entities SET present=1, track_id=?, last_seen=? WHERE id=?",
                (track_id, time.time(), eid),
            )
        self.db.commit()
        self.present[eid] = {
            "name": name, "track_id": track_id, "since": time.time(), "attending": 0.0
        }
        return eid, name, identity

    def person_left(self, track_id: str) -> Optional[str]:
        for eid, info in list(self.present.items()):
            if info["track_id"] == track_id:
                del self.present[eid]
                self.db.execute(
                    "UPDATE entities SET present=0, last_seen=? WHERE id=?",
                    (time.time(), eid),
                )
                self.db.commit()
                return eid
        return None

    def match_face(self, embedding: list[float]) -> Optional[tuple[str, str, str]]:
        """cosine vs known_faces -> (entity_id, name, known|uncertain) | None"""
        from lucas_common import config

        rows = self.db.execute("SELECT entity_id, name, embedding FROM known_faces").fetchall()
        if not rows:
            return None
        q = np.asarray(embedding, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        best, best_sim = None, -1.0
        for eid, name, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            sim = float(q @ (v / (np.linalg.norm(v) + 1e-9)))
            if sim > best_sim:
                best, best_sim = (eid, name), sim
        known_t = config.get("node_a.identity.known_threshold", 0.55)
        uncertain_t = config.get("node_a.identity.uncertain_threshold", 0.40)
        if best_sim >= known_t:
            return best[0], best[1], "known"
        if best_sim >= uncertain_t:
            return best[0], best[1], "uncertain"
        return None

    def enroll_face(self, name: str, embedding: list[float]) -> str:
        eid = new_id()
        vec = np.asarray(embedding, dtype=np.float32).tobytes()
        with self.db:
            self.db.execute(
                "INSERT INTO entities(id, kind, name, first_seen, last_seen)"
                " VALUES(?, 'person', ?, ?, ?)", (eid, name, time.time(), time.time()))
            self.db.execute(
                "INSERT INTO known_faces(entity_id, name, embedding) VALUES(?,?,?)",
                (eid, name, vec))
        return eid

    # ------------------------------------------------------------ events

    def add_event(self, type_: str, description: str,
                  entity_ids: list[str] | None = None, salience: float = 0.3) -> str:
        eid = new_id()
        self.db.execute(
            "INSERT INTO events(id, ts, type, description, entity_ids, salience)"
            " VALUES(?,?,?,?,?,?)",
            (eid, time.time(), type_, description, json.dumps(entity_ids or []), salience),
        )
        self.db.commit()
        return eid

    def recent_events(self, k: int = 5) -> list[tuple[float, str]]:
        return self.db.execute(
            "SELECT ts, description FROM events ORDER BY ts DESC LIMIT ?", (k,)
        ).fetchall()[::-1]

    # ------------------------------------------------------------- facts

    def commit_fact(self, statement: str, source: str, confidence: float = 0.7) -> str:
        """Reconciliation-lite: exact-ish dedupe now; embedding dedupe via Node C later."""
        row = self.db.execute(
            "SELECT id, confidence FROM facts WHERE statement=? AND invalidated IS NULL",
            (statement,),
        ).fetchone()
        if row:
            fid, conf = row
            self.db.execute(
                "UPDATE facts SET confidence=? WHERE id=?",
                (min(0.99, conf + (1 - conf) * 0.3), fid),
            )
            self.db.commit()
            return fid
        fid = new_id()
        self.db.execute(
            "INSERT INTO facts(id, statement, confidence, source, created) VALUES(?,?,?,?,?)",
            (fid, statement, confidence, source, time.time()),
        )
        self.db.commit()
        return fid

    def facts_for_prompt(self, k: int = 3) -> list[str]:
        rows = self.db.execute(
            "SELECT statement FROM facts WHERE invalidated IS NULL"
            " ORDER BY confidence DESC, created DESC LIMIT ?", (k,)).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------- trace

    def trace(self, deliberation_id: str, kind: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO trace(ts, deliberation_id, kind, payload) VALUES(?,?,?,?)",
            (time.time(), deliberation_id, kind, json.dumps(payload)[:4000]),
        )
        self.db.commit()
