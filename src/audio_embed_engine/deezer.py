"""Deezer lookup: track metadata -> fresh 30 s preview URL, within Deezer's rate limit.

Deezer allows ~50 API calls per 5 s per IP. Exceeding it returns HTTP 200 with
`{"error": {"code": 4, "message": "Quota limit exceeded"}}` (measured 2026-11-06),
so we throttle client-side and back off a full window if we still trip it.
Preview MP3s are served from a CDN that is not counted against that quota.
"""

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .types import Track

API = "https://api.deezer.com"
QUOTA_ERROR = 4
NOT_FOUND_ERROR = 800
DURATION_TOLERANCE_S = 3

type Json = dict[str, Any]


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` acquisitions per `window` seconds."""

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = window
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.window - (now - self._calls[0])
            time.sleep(wait)

    def penalize(self) -> None:
        """Called after a quota error: treat the window as full from now."""
        with self._lock:
            now = time.monotonic()
            self._calls = deque([now] * self.max_calls)


@dataclass(frozen=True)
class Preview:
    deezer_id: int
    url: str  # signed and short-lived: use immediately, never persist


type Resolution = Preview | Literal["not_found", "no_preview"]


class DeezerError(Exception):
    """Transient failure (network, 5xx, persistent quota errors) after retries."""


class Deezer:
    def __init__(
        self,
        client: httpx.Client | None = None,
        max_calls: int = 45,  # headroom under the measured ~50 / 5 s
        window: float = 5.0,
        retries: int = 4,
    ) -> None:
        self.client = client or httpx.Client(timeout=20, follow_redirects=True)
        self.limiter = RateLimiter(max_calls, window)
        self.retries = retries
        self.api_calls = 0

    def _api(self, path: str, params: dict[str, str | int] | None = None) -> Json | None:
        """GET an API path. Returns None for Deezer's 'no data' error, raises DeezerError when retries run out."""
        for attempt in range(self.retries + 1):
            self.limiter.acquire()
            self.api_calls += 1
            try:
                r = self.client.get(API + path, params=params)
                if r.status_code >= 500 or r.status_code == 429:
                    raise DeezerError(f"HTTP {r.status_code}")
                body: Json = r.json()
            except (httpx.HTTPError, ValueError, DeezerError) as e:
                if attempt == self.retries:
                    raise DeezerError(f"{path}: {e}") from e
                time.sleep(0.5 * 2**attempt)
                continue
            error = body.get("error")
            if not error:
                return body
            if error.get("code") == NOT_FOUND_ERROR:
                return None
            if error.get("code") == QUOTA_ERROR and attempt < self.retries:
                self.limiter.penalize()
                continue
            raise DeezerError(f"{path}: {error}")
        raise AssertionError("unreachable")

    def resolve(self, track: Track) -> Resolution:
        """Find a playable preview: ISRC lookup first, then search on artist + title matched by duration."""
        found_without_preview = False
        if track.isrc:
            hit = self._api(f"/track/isrc:{track.isrc}")
            if hit and hit.get("preview"):
                return Preview(hit["id"], hit["preview"])
            found_without_preview = hit is not None
        for query in _search_queries(track):
            body = self._api("/search", {"q": query, "limit": 10})
            for cand in (body or {}).get("data", []):
                if not _matches(track, cand):
                    continue
                if cand.get("preview"):
                    return Preview(cand["id"], cand["preview"])
                found_without_preview = True
        return "no_preview" if found_without_preview else "not_found"

    def download(self, preview: Preview) -> bytes:
        """Fetch the preview MP3 (CDN, not rate limited). Retries transient failures."""
        for attempt in range(self.retries + 1):
            try:
                r = self.client.get(preview.url)
                r.raise_for_status()
                return r.content
            except httpx.HTTPError as e:
                if attempt == self.retries:
                    raise DeezerError(f"download {preview.deezer_id}: {e}") from e
                time.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")


def _norm(s: str) -> str:
    return re.sub(r"[^\w]", "", s.casefold())


def _clean_title(title: str) -> str:
    """'Song - Remastered 2011 (feat. X)' -> 'Song'."""
    title = re.sub(r"\s*[\(\[](feat|with|ft)\.?[^\)\]]*[\)\]]", "", title, flags=re.IGNORECASE)
    return re.split(r"\s+-\s+", title)[0].strip()


def _search_queries(track: Track) -> list[str]:
    # Deezer's advanced `artist:"…" track:"…"` syntax returned nothing in testing; plain queries work.
    artist = track.artists[0] if track.artists else ""
    queries = [f"{artist} {track.title}".strip()]
    cleaned = f"{artist} {_clean_title(track.title)}".strip()
    if cleaned != queries[0]:
        queries.append(cleaned)
    return queries


def _matches(track: Track, cand: Json) -> bool:
    if abs(cand.get("duration", -999) - track.duration_ms / 1000) > DURATION_TOLERANCE_S:
        return False
    cand_artist = _norm(cand.get("artist", {}).get("name", ""))
    if not cand_artist:
        return False
    return any(n and (n in cand_artist or cand_artist in n) for n in map(_norm, track.artists))
