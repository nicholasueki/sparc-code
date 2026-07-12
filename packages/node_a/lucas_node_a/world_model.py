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
        # face embeddings buffered per entity, SURVIVES presence churn so a
        # flapping tracker can't wipe enrollment samples (entity_id -> [emb,...])
        self.face_buffer: dict[str, list] = {}
        self._rebuild_hot()

    def entity_by_track(self, track_id: str) -> Optional[str]:
        for eid, info in self.present.items():
            if info.get("track_id") == track_id:
                return eid
        return None

    def buffer_face(self, entity_id: str, embedding: list[float], cap: int = 12) -> int:
        buf = self.face_buffer.setdefault(entity_id, [])
        buf.append(embedding)
        del buf[:-cap]
        return len(buf)

    def face_samples(self, entity_id: str) -> list[list[float]]:
        return self.face_buffer.get(entity_id, [])

    def _rebuild_hot(self) -> None:
        for eid, name, track_id, last_seen in self.db.execute(
            "SELECT id, name, track_id, last_seen FROM entities WHERE present=1"
        ):
            # anything 'present' at boot is stale — mark absent, real presence re-derives
            self.db.execute("UPDATE entities SET present=0 WHERE id=?", (eid,))
        self.db.commit()

    # ---------------------------------------------------------- presence

    REAPPEAR_WINDOW_S = 30.0  # object permanence: brief absence != new person

    def person_appeared(self, track_id: str, embedding: Optional[list[float]] = None
                        ) -> tuple[str, Optional[str], str, bool]:
        """-> (entity_id, name|None, identity: known|unknown|uncertain, is_reappearance)

        Deterministic object permanence (v2 §2.4): without face identity, a person
        entity that went absent moments ago and a new track appearing shortly after
        are treated as the same person — tracker flaps must not mint new people."""
        name, identity, eid = None, "unknown", None
        if embedding is not None:
            match = self.match_face(embedding)
            if match:
                eid, name, identity = match
        reappeared = False
        if eid is None:
            row = self.db.execute(
                "SELECT id, name FROM entities WHERE kind='person' AND present=0"
                " AND last_seen > ? ORDER BY last_seen DESC LIMIT 1",
                (time.time() - self.REAPPEAR_WINDOW_S,),
            ).fetchone()
            if row:
                eid, name = row
                identity = "reappeared"
                reappeared = True
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
        return eid, name, identity, reappeared

    def enroll_present(self, entity_id: str, name: str) -> Optional[str]:
        """Enroll the currently-present entity under `name`, using the rolling
        embedding buffer the enrichment loop has been filling. Deterministic;
        called only after the validator approves an enroll_face action."""
        info = self.present.get(entity_id)
        embs = self.face_samples(entity_id)
        if info is None or len(embs) < 3:
            return None
        emb = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        sims = [float(np.asarray(e) @ emb / (np.linalg.norm(e) + 1e-9)) for e in embs]
        if min(sims) < 0.5:  # inconsistent samples: possibly two faces — refuse
            return None
        with self.db:
            self.db.execute("UPDATE entities SET name=? WHERE id=?", (name, entity_id))
            self.db.execute(
                "INSERT OR REPLACE INTO known_faces(entity_id, name, embedding)"
                " VALUES(?,?,?)", (entity_id, name, emb.astype(np.float32).tobytes()))
        info["name"] = name
        return entity_id

    def known_names(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT name FROM known_faces")]

    def update_identity(self, entity_id: str, embedding: list[float]
                        ) -> tuple[str, Optional[str], str]:
        """Resolve a live entity's identity from a face embedding.

        -> (entity_id_after_merge, name|None, known|uncertain|unknown).
        If the embedding matches an enrolled person, the temporary entity is
        merged into the known one (presence + track transfer)."""
        info = self.present.get(entity_id)
        if info is None:
            return entity_id, None, "unknown"
        if info.get("name"):
            return entity_id, info["name"], "known"
        match = self.match_face(embedding)
        if match is None:
            return entity_id, None, "unknown"
        known_eid, name, quality = match
        if quality != "known":
            return entity_id, None, quality
        if known_eid != entity_id:  # merge temp entity into the enrolled one
            self.present.pop(entity_id, None)
            self.present[known_eid] = {**info, "name": name}
            with self.db:
                self.db.execute("UPDATE entities SET present=0 WHERE id=?", (entity_id,))
                self.db.execute(
                    "UPDATE entities SET present=1, track_id=?, last_seen=? WHERE id=?",
                    (info["track_id"], time.time(), known_eid))
        else:
            info["name"] = name
        return known_eid, name, "known"

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
            # last_seen=0: enrollment must never trigger the reappearance
            # shortcut — identity comes from the face match, not a timestamp
            self.db.execute(
                "INSERT INTO entities(id, kind, name, first_seen, last_seen)"
                " VALUES(?, 'person', ?, ?, 0)", (eid, name, time.time()))
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

    def recent_events(self, k: int = 5) -> list[tuple[float, str, str]]:
        """-> [(ts, type, description)] oldest-first."""
        return self.db.execute(
            "SELECT ts, type, description FROM events ORDER BY ts DESC LIMIT ?", (k,)
        ).fetchall()[::-1]

    # events that just track presence — the "Present:" line already conveys this,
    # so they're clutter (and anonymous "they") in the scene's Recently list.
    _PRESENCE_TYPES = ("person_entered", "person_left")

    def recent_notable_events(self, k: int = 3) -> list[tuple[float, str, str]]:
        """Notable = non-presence events, consecutive duplicates collapsed. Oldest-first."""
        rows = self.db.execute(
            "SELECT ts, type, description FROM events "
            f"WHERE type NOT IN ({','.join('?' * len(self._PRESENCE_TYPES))}) "
            "ORDER BY ts DESC LIMIT ?", (*self._PRESENCE_TYPES, k * 4)).fetchall()
        out: list[tuple[float, str, str]] = []
        last_desc = None
        for ts, t, d in rows:  # newest first
            if d == last_desc:
                continue
            last_desc = d
            out.append((ts, t, d))
            if len(out) >= k:
                break
        return out[::-1]

    # ------------------------------------------------------------- facts

    def commit_fact(self, statement: str, source: str, confidence: float = 0.7,
                    similar: tuple[str, float] | None = None,
                    dup_threshold: float = 0.92) -> str:
        """Reconciliation: exact dedupe here; semantic dedupe via `similar` =
        (existing_fact_id, cosine_score) supplied by the caller's vector lookup."""
        if similar and similar[1] >= dup_threshold:
            fid = similar[0]
            row = self.db.execute(
                "SELECT confidence FROM facts WHERE id=? AND invalidated IS NULL", (fid,)
            ).fetchone()
            if row:
                self.db.execute(
                    "UPDATE facts SET confidence=? WHERE id=?",
                    (min(0.99, row[0] + (1 - row[0]) * 0.3), fid),
                )
                self.db.commit()
                return fid
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
