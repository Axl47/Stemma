"""Persistent results cache: one SQLite row per (track key, model id).

Successful rows hold the L2-normalised float32 vector (+ optional tags). Failed rows hold the
status and a `retry_after` timestamp so we don't re-ask Deezer for tracks it doesn't have.
A new model id simply adds rows next to the old ones; nothing is invalidated.
"""

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Status

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    track_key   TEXT NOT NULL,   -- ISRC (uppercase) or the caller's key when there is no ISRC
    model       TEXT NOT NULL,   -- model id incl. version, see models.py
    status      TEXT NOT NULL,   -- 'ok' or a failure status
    vector      BLOB,            -- float32 little-endian, L2-normalised; NULL on failure
    tags        TEXT,            -- JSON {"styles": {name: score}, "moods": {...}}; optional
    deezer_id   INTEGER,
    detail      TEXT,            -- failure message
    retry_after REAL,            -- unix seconds; failures become eligible again after this
    updated_at  REAL NOT NULL,
    PRIMARY KEY (track_key, model)
) WITHOUT ROWID;
"""

# How long a failure is trusted before we try again.
RETRY_AFTER_S: dict[Status, float] = {
    "not_found": 14 * 86400,
    "no_preview": 14 * 86400,
    "decode_error": 7 * 86400,
    "download_error": 3600,
    "error": 3600,
}

type Tags = dict[str, dict[str, float]]  # {"styles": {name: score}, "moods": {name: score}}, top-k only


@dataclass(frozen=True)
class Row:
    track_key: str
    status: Status
    vector: np.ndarray | None
    tags: Tags | None
    deezer_id: int | None
    detail: str | None


class Cache:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def pending(self, keys: Iterable[str], model: str, retry_failed: bool = False) -> set[str]:
        """Keys with no row for `model`, or a failure whose retry time has passed (or any failure if retry_failed)."""
        now = time.time()
        done = {
            key
            for key, status, retry_after in self.db.execute(
                "SELECT track_key, status, retry_after FROM embeddings WHERE model = ?", (model,)
            )
            if status == "ok" or (not retry_failed and (retry_after or 0) > now)
        }
        return set(keys) - done

    def put_ok(self, key: str, model: str, vector: np.ndarray, tags: Tags | None, deezer_id: int) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, 'ok', ?, ?, ?, NULL, NULL, ?)",
            (key, model, vector.astype("<f4").tobytes(), json.dumps(tags) if tags else None, deezer_id, time.time()),
        )

    def put_failure(self, key: str, model: str, status: Status, detail: str = "", deezer_id: int | None = None) -> None:
        now = time.time()
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?)",
            (key, model, status, deezer_id, detail[:500], now + RETRY_AFTER_S[status], now),
        )

    def commit(self) -> None:
        self.db.commit()

    def get_many(self, keys: Iterable[str], model: str) -> dict[str, Row]:
        wanted = set(keys)
        rows = self.db.execute(
            "SELECT track_key, status, vector, tags, deezer_id, detail FROM embeddings WHERE model = ?", (model,)
        )
        return {
            key: Row(
                key, status, np.frombuffer(vec, "<f4") if vec else None, json.loads(tags) if tags else None, did, detail
            )
            for key, status, vec, tags, did, detail in rows
            if key in wanted
        }
