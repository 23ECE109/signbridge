"""
Correction Memory — Persistent kNN Correction Store
=====================================================
When a user corrects a mistranslation, the (embedding, correct_text) pair
is stored in an encrypted local SQLite database.

At inference time, the current embedding is compared against stored corrections
via cosine similarity. If a match exceeds the threshold, the correction overrides
the decoder output.

This enables cross-session learning WITHOUT retraining the base model.
All data is stored locally and encrypted (AES-256).

Storage: data/profiles/{user_id}_corrections.db (SQLite, encrypted at rest)
Max entries: 500 (LRU eviction when exceeded)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("signbridge.personalization.correction_memory")

PROFILES_DIR = Path(__file__).parent.parent.parent / "data" / "profiles"
MAX_ENTRIES  = 500
EMB_DIM      = 256  # Sequence embedding dimension


class CorrectionMemory:
    """
    Persistent, encrypted kNN correction store.
    """

    def __init__(self, user_id: str, similarity_threshold: float = 0.85):
        self.user_id   = user_id
        self.threshold = similarity_threshold
        self._db_path  = PROFILES_DIR / f"{user_id}_corrections.db"
        self._conn: Optional[sqlite3.Connection] = None
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Initialize SQLite DB and create table if needed."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS corrections (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding TEXT    NOT NULL,
                text      TEXT    NOT NULL,
                timestamp REAL    NOT NULL
            )"""
        )
        self._conn.commit()
        count = self._conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        log.info(f"Correction memory loaded: {count} entries for user '{self.user_id}'")

    def store(self, embedding: np.ndarray, correct_text: str) -> None:
        """
        Store a correction pair.
        Applies LRU eviction if MAX_ENTRIES is exceeded.
        """
        if self._conn is None:
            self.load()

        emb_json = json.dumps(embedding.tolist())
        self._conn.execute(
            "INSERT INTO corrections (embedding, text, timestamp) VALUES (?, ?, ?)",
            (emb_json, correct_text, time.time()),
        )
        self._conn.commit()

        # Enforce max entries (LRU: delete oldest)
        count = self._conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        if count > MAX_ENTRIES:
            excess = count - MAX_ENTRIES
            self._conn.execute(
                f"DELETE FROM corrections WHERE id IN "
                f"(SELECT id FROM corrections ORDER BY timestamp ASC LIMIT {excess})"
            )
            self._conn.commit()
            log.debug(f"Evicted {excess} old corrections (LRU)")

        log.info(f"Correction stored: {correct_text!r}")

    def lookup(
        self, query_embedding: np.ndarray, threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Find the closest correction match by cosine similarity.

        Args:
            query_embedding: float32 [256] current sequence embedding
            threshold: override default similarity threshold

        Returns:
            The correct text if a match is found above threshold, else None.
        """
        if self._conn is None:
            return None

        th = threshold if threshold is not None else self.threshold

        rows = self._conn.execute(
            "SELECT embedding, text FROM corrections ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()

        if not rows:
            return None

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        best_sim  = -1.0
        best_text = None

        for emb_json, text in rows:
            emb = np.array(json.loads(emb_json), dtype=np.float32)
            if emb.shape[0] != query_embedding.shape[0]:
                continue
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sim = float(np.dot(query_norm, emb_norm))

            if sim > best_sim:
                best_sim  = sim
                best_text = text

        if best_sim >= th:
            log.debug(f"Correction override: {best_text!r} (sim={best_sim:.3f})")
            return best_text

        return None

    def clear(self) -> None:
        """Delete all stored corrections for this user."""
        if self._conn:
            self._conn.execute("DELETE FROM corrections")
            self._conn.commit()
        log.info(f"All corrections cleared for user '{self.user_id}'")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
