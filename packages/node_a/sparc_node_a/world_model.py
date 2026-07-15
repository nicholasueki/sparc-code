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

from sparc_common.types import new_id

log = logging.getLogger("sparc.world")

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
        # Contradictory evidence is deliberately live-only.  It must never alter
        # a durable enrollment until enough mutually-consistent face samples have
        # established a replacement presentation.
        self.contradiction_buffer: dict[str, list[dict]] = {}
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
        """Create a live presentation from face evidence or anonymous continuity.

        Timestamp proximity can only reuse an anonymous entity.  An enrolled
        entity becomes live only after a positive face match.
        """
        name, identity, provenance, eid = None, "unknown", "none", None
        ownership_conflict = None
        if embedding is not None:
            evidence = self._face_evidence(embedding)
            if evidence and evidence[2] == "known":
                candidate_eid, candidate_name, _, similarity = evidence
                incumbent = self.present.get(candidate_eid)
                if (incumbent is not None
                        and incumbent.get("track_id") != track_id):
                    identity, provenance = "uncertain", "face"
                    ownership_conflict = (candidate_eid, similarity)
                else:
                    eid, name, identity = candidate_eid, candidate_name, "known"
                    provenance = "face"
                    self._trace_identity(
                        "confirm", "known_match", None, eid, eid, similarity, 1)
            elif evidence and evidence[2] == "uncertain":
                identity, provenance = "uncertain", "face"
        reappeared = False
        if eid is None and embedding is None:
            row = self.db.execute(
                "SELECT id FROM entities WHERE kind='person' AND present=0"
                " AND name IS NULL"
                " AND last_seen > ? ORDER BY last_seen DESC LIMIT 1",
                (time.time() - self.REAPPEAR_WINDOW_S,),
            ).fetchone()
            if row:
                eid = row[0]
                identity, provenance = "unknown", "timestamp"
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
            "name": name,
            "track_id": track_id,
            "since": time.time(),
            "attending": 0.0,
            "identity_state": identity,
            "identity_provenance": provenance,
        }
        if identity == "uncertain":
            evidence = self._face_evidence(embedding or [])
            if ownership_conflict:
                self._trace_identity(
                    "ownership_conflict", "known_match_live_conflict",
                    eid, eid, ownership_conflict[0], ownership_conflict[1], 1)
            else:
                self._trace_identity(
                    "uncertain", "uncertain_match", None, eid,
                    evidence[0] if evidence else None,
                    evidence[3] if evidence else None, 1)
        return eid, name, identity, reappeared

    def live_name(self, entity_id: str) -> Optional[str]:
        """A name safe for scene text and greetings, or ``None``.

        The durable name is intentionally retained during contradiction handling;
        presentation code must use this gate instead of reading ``name`` directly.
        """
        info = self.present.get(entity_id)
        if (info and info.get("identity_state") == "known"
                and info.get("identity_provenance") == "face"):
            return info.get("name")
        return None

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
        info["identity_state"] = "known"
        info["identity_provenance"] = "face"
        self.contradiction_buffer.pop(entity_id, None)
        return entity_id

    def known_names(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT name FROM known_faces")]

    def update_identity(self, entity_id: str, embedding: list[float],
                        detection_confidence: float = 1.0
                        ) -> tuple[str, Optional[str], str]:
        """Apply one face sample to the authoritative live identity state."""
        info = self.present.get(entity_id)
        if info is None:
            return entity_id, None, "unknown"
        evidence = self._face_evidence(embedding)
        durable_known = bool(info.get("name"))

        if not durable_known:
            self.contradiction_buffer.pop(entity_id, None)
            if evidence is None or evidence[2] == "negative":
                info["identity_state"] = "unknown"
                info["identity_provenance"] = "face"
                return entity_id, None, "unknown"
            known_eid, name, quality, similarity = evidence
            if quality == "uncertain":
                info["identity_state"] = "uncertain"
                info["identity_provenance"] = "face"
                self._trace_identity(
                    "uncertain", "uncertain_match", entity_id, entity_id,
                    known_eid, similarity, 1)
                return entity_id, None, "uncertain"
            return self._bind_anonymous(entity_id, known_eid, name, similarity)

        if evidence and evidence[2] == "known" and evidence[0] == entity_id:
            info["identity_state"] = "known"
            info["identity_provenance"] = "face"
            self.contradiction_buffer.pop(entity_id, None)
            self._trace_identity(
                "confirm", "known_match", entity_id, entity_id,
                entity_id, evidence[3], 1)
            return entity_id, info["name"], "known"

        # Any contradiction immediately suppresses the durable name.  The
        # enrollment remains untouched while the evidence buffer resolves it.
        info["identity_state"] = "uncertain"
        info["identity_provenance"] = "face"
        if evidence and evidence[2] == "known":
            candidate, name, _, similarity = evidence
            samples = self._append_consistent_evidence(
                entity_id, "different_known", candidate, embedding,
                detection_confidence)
            if len(samples) >= int(self._identity_config(
                    "different_known_samples", 2)):
                return self._rebind_known(entity_id, candidate, name, similarity)
            self._trace_identity(
                "uncertain", "different_known_pending", entity_id, entity_id,
                candidate, similarity, len(samples))
            return entity_id, None, "uncertain"

        if evidence and evidence[2] == "uncertain":
            self.contradiction_buffer.pop(entity_id, None)
            self._trace_identity(
                "uncertain", "uncertain_match", entity_id, entity_id,
                evidence[0], evidence[3], 1)
            return entity_id, None, "uncertain"

        samples = self._append_consistent_evidence(
            entity_id, "strong_negative", None, embedding,
            detection_confidence)
        if len(samples) >= int(self._identity_config(
                "strong_negative_samples", 3)):
            return self._split_known(entity_id, samples)
        self._trace_identity(
            "uncertain", "strong_negative", entity_id, entity_id,
            None, evidence[3] if evidence else None, len(samples))
        return entity_id, None, "uncertain"

    def _bind_anonymous(self, entity_id: str, known_eid: str, name: str,
                        similarity: float) -> tuple[str, Optional[str], str]:
        info = self.present[entity_id]
        incumbent = self.present.get(known_eid)
        if (incumbent is not None
                and incumbent.get("track_id") != info.get("track_id")):
            info["identity_state"] = "uncertain"
            info["identity_provenance"] = "face"
            self._trace_identity(
                "ownership_conflict", "known_match_live_conflict",
                entity_id, entity_id, known_eid, similarity, 1)
            return entity_id, None, "uncertain"
        info = self.present.pop(entity_id)
        self.present[known_eid] = {
            **info, "name": name, "identity_state": "known",
            "identity_provenance": "face",
        }
        with self.db:
            self.db.execute("UPDATE entities SET present=0 WHERE id=?", (entity_id,))
            self.db.execute(
                "UPDATE entities SET present=1, track_id=?, last_seen=? WHERE id=?",
                (info["track_id"], time.time(), known_eid))
        self.contradiction_buffer.pop(entity_id, None)
        self._trace_identity(
            "confirm", "known_match", entity_id, known_eid,
            known_eid, similarity, 1)
        return known_eid, name, "known"

    def _rebind_known(self, entity_id: str, known_eid: str, name: str,
                      similarity: float) -> tuple[str, Optional[str], str]:
        info = self.present[entity_id]
        count = len(self.contradiction_buffer.get(entity_id, []))
        incumbent = self.present.get(known_eid)
        if (incumbent is not None
                and incumbent.get("track_id") != info.get("track_id")):
            info["identity_state"] = "uncertain"
            info["identity_provenance"] = "face"
            self._trace_identity(
                "ownership_conflict", "different_known_live_conflict",
                entity_id, entity_id, known_eid, similarity, count)
            return entity_id, None, "uncertain"
        info = self.present.pop(entity_id)
        self.present[known_eid] = {
            **info, "name": name, "identity_state": "known",
            "identity_provenance": "face",
        }
        with self.db:
            self.db.execute(
                "UPDATE entities SET present=0, last_seen=? WHERE id=?",
                (time.time(), entity_id))
            self.db.execute(
                "UPDATE entities SET present=1, track_id=?, last_seen=? WHERE id=?",
                (info["track_id"], time.time(), known_eid))
        self.contradiction_buffer.pop(entity_id, None)
        self.contradiction_buffer.pop(known_eid, None)
        self._trace_identity(
            "rebind", "different_known", entity_id, known_eid,
            known_eid, similarity, count)
        return known_eid, name, "known"

    def _split_known(self, entity_id: str,
                     samples: list[dict]) -> tuple[str, None, str]:
        info = self.present.pop(entity_id)
        new_eid = new_id()
        now = time.time()
        with self.db:
            self.db.execute(
                "UPDATE entities SET present=0, last_seen=? WHERE id=?",
                (now, entity_id))
            self.db.execute(
                "INSERT INTO entities(id, kind, track_id, present, first_seen, last_seen)"
                " VALUES(?, 'person', ?, 1, ?, ?)",
                (new_eid, info["track_id"], info["since"], now))
        self.present[new_eid] = {
            **info, "name": None, "identity_state": "unknown",
            "identity_provenance": "face",
        }
        # Only the mutually-consistent samples that established the correction
        # follow the live track.  Durable enrollment samples stay with the known
        # entity.
        self.face_buffer[new_eid] = [s["embedding"] for s in samples]
        self.contradiction_buffer.pop(entity_id, None)
        self._trace_identity(
            "split", "strong_negative", entity_id, new_eid,
            None, None, len(samples))
        return new_eid, None, "unknown"

    def _append_consistent_evidence(
            self, entity_id: str, evidence_class: str,
            candidate_id: Optional[str], embedding: list[float],
            detection_confidence: float) -> list[dict]:
        min_conf = float(self._identity_config(
            "min_face_confidence",
            self._config_get("node_a.face.face_conf", 0.55)))
        if detection_confidence < min_conf:
            self.contradiction_buffer.pop(entity_id, None)
            return []
        sample = {
            "class": evidence_class,
            "candidate_id": candidate_id,
            "embedding": embedding,
            "confidence": detection_confidence,
        }
        existing = self.contradiction_buffer.get(entity_id, [])
        if (not existing
                or existing[0]["class"] != evidence_class
                or existing[0]["candidate_id"] != candidate_id
                or not self._embedding_agrees(embedding, existing)):
            existing = [sample]
        else:
            existing.append(sample)
        required = max(
            int(self._identity_config("different_known_samples", 2)),
            int(self._identity_config("strong_negative_samples", 3)),
        )
        cap = max(1, int(self._identity_config(
            "contradiction_buffer_cap", required)))
        existing = existing[-cap:]
        self.contradiction_buffer[entity_id] = existing
        return existing

    def _embedding_agrees(self, embedding: list[float], samples: list[dict]) -> bool:
        threshold = float(self._identity_config("evidence_agreement_threshold", 0.80))
        q = np.asarray(embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        for sample in samples:
            v = np.asarray(sample["embedding"], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-9
            if float(q @ v) < threshold:
                return False
        return True

    @staticmethod
    def _config_get(path: str, default):
        from sparc_common import config
        return config.get(path, default)

    def _identity_config(self, key: str, default):
        return self._config_get(f"node_a.identity.{key}", default)

    def person_left(self, track_id: str) -> Optional[str]:
        for eid, info in list(self.present.items()):
            if info["track_id"] == track_id:
                del self.present[eid]
                self.db.execute(
                    "UPDATE entities SET present=0, last_seen=? WHERE id=?",
                    (time.time(), eid),
                )
                self.db.commit()
                self.contradiction_buffer.pop(eid, None)
                return eid
        return None

    def match_face(self, embedding: list[float]) -> Optional[tuple[str, str, str]]:
        """cosine vs known_faces -> (entity_id, name, known|uncertain) | None"""
        evidence = self._face_evidence(embedding)
        if evidence is None or evidence[2] == "negative":
            return None
        return evidence[:3]

    def _face_evidence(
            self, embedding: list[float]
            ) -> Optional[tuple[str, str, str, float]]:
        """Return the closest enrollment plus its evidence class and cosine."""
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
        known_t = float(self._identity_config("known_threshold", 0.55))
        uncertain_t = float(self._identity_config("uncertain_threshold", 0.40))
        if best_sim >= known_t:
            return best[0], best[1], "known", best_sim
        if best_sim >= uncertain_t:
            return best[0], best[1], "uncertain", best_sim
        return best[0], best[1], "negative", best_sim

    def _trace_identity(
            self, transition: str, evidence_class: str,
            from_entity: Optional[str], to_entity: Optional[str],
            candidate_entity: Optional[str], similarity: Optional[float],
            sample_count: int) -> None:
        self.trace("-", "identity_transition", {
            "transition": transition,
            "evidence_class": evidence_class,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "candidate_entity": candidate_entity,
            "similarity": None if similarity is None else round(similarity, 4),
            "sample_count": sample_count,
        })

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
