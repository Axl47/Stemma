"""Shared data types: the input track record and per-track outcomes."""

import json
from dataclasses import dataclass
from typing import Literal

# `ok` rows carry a vector; every other status is a cached failure with a retry-after time.
type Status = Literal["ok", "not_found", "no_preview", "download_error", "decode_error", "error"]
FAILURE_STATUSES: tuple[Status, ...] = ("not_found", "no_preview", "download_error", "decode_error", "error")


@dataclass(frozen=True)
class Track:
    """One input line: `{isrc, title, artists, durationMs, key?}`. Unknown fields are ignored."""

    isrc: str | None
    title: str
    artists: tuple[str, ...]
    duration_ms: int
    key: str | None = None

    @property
    def cache_key(self) -> str:
        """ISRC when known (shared across releases of one recording), else the caller's `key`."""
        if self.isrc:
            return self.isrc.upper()
        if self.key:
            return self.key
        raise ValueError(f"track {self.title!r} has neither isrc nor key")

    @classmethod
    def from_json(cls, line: str) -> Track:
        d = json.loads(line)
        return cls(
            isrc=d.get("isrc") or None,
            title=d["title"],
            artists=tuple(d.get("artists") or ()),
            duration_ms=int(d["durationMs"]),
            key=d.get("key") or None,
        )


def read_tracks(path: str) -> list[Track]:
    with open(path, encoding="utf-8") as f:
        return [Track.from_json(line) for line in f if line.strip()]
