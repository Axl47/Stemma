"""The pipeline: tracks -> (cache check) -> Deezer resolve + download + decode (threads) -> batched inference -> cache.

Fetching is I/O- and rate-limit-bound, so it runs on a thread pool; inference runs on the
calling thread in batches as decoded audio arrives (onnxruntime releases the GIL, so both
overlap). Only the calling thread touches SQLite. Audio lives in memory only until embedded.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .audio import Audio, DecodeError, decode
from .cache import Cache, Tags
from .deezer import Deezer, DeezerError, Preview
from .models import Model
from .types import Status, Track


@dataclass
class EmbedStats:
    total: int = 0  # unique cache keys in the input
    cached: int = 0  # already done (or failed recently) for this model
    statuses: Counter[Status] = field(default_factory=Counter)  # outcomes of this run
    api_calls: int = 0
    fetch_s: float = 0.0  # summed per-track resolve + download + decode time (across threads)
    inference_s: float = 0.0  # time spent in model.embed
    wall_s: float = 0.0


@dataclass(frozen=True)
class _Fetched:
    track: Track
    audio: Audio
    deezer_id: int


@dataclass(frozen=True)
class _Failed:
    track: Track
    status: Status
    detail: str = ""


def _fetch(dz: Deezer, track: Track, sample_rate: int) -> tuple[_Fetched | _Failed, float]:
    start = time.perf_counter()
    result: _Fetched | _Failed
    try:
        res = dz.resolve(track)
        if isinstance(res, Preview):
            try:
                result = _Fetched(track, decode(dz.download(res), sample_rate), res.deezer_id)
            except DecodeError as e:
                result = _Failed(track, "decode_error", str(e))
        else:
            result = _Failed(track, res)
    except DeezerError as e:
        result = _Failed(track, "download_error", str(e))
    except Exception as e:  # noqa: BLE001 -- never let one track kill a bulk run; it is cached as `error` and retried later
        result = _Failed(track, "error", f"{type(e).__name__}: {e}")
    return result, time.perf_counter() - start


def embed_tracks(
    tracks: Iterable[Track],
    cache: Cache,
    model: Model,
    deezer: Deezer | None = None,
    workers: int = 16,
    batch_size: int = 32,
    retry_failed: bool = False,
    on_progress: Callable[[EmbedStats], None] | None = None,
) -> EmbedStats:
    """Embed every track not already cached for `model.id`; returns what happened.
    Cached rows (ok, or failed with a retry time still in the future) are skipped without any I/O."""
    t0 = time.perf_counter()
    by_key = {t.cache_key: t for t in tracks}
    stats = EmbedStats(total=len(by_key))
    model_id = model.id
    todo = [by_key[k] for k in sorted(cache.pending(by_key, model_id, retry_failed))]
    stats.cached = stats.total - len(todo)
    if not todo:
        stats.wall_s = time.perf_counter() - t0
        return stats

    dz = deezer or Deezer()
    batch: list[_Fetched] = []

    def flush() -> None:
        if not batch:
            return
        t = time.perf_counter()
        embs = model.embed([f.audio for f in batch])
        stats.inference_s += time.perf_counter() - t
        for f, e in zip(batch, embs, strict=True):
            cache.put_ok(f.track.cache_key, model_id, e.vector, e.tags, f.deezer_id)
        stats.statuses["ok"] += len(batch)
        cache.commit()
        batch.clear()
        if on_progress:
            on_progress(stats)

    queue = iter(todo)
    with ThreadPoolExecutor(workers) as pool:
        # Bounded in-flight window so decoded audio can't pile up faster than inference drains it.
        inflight: set[Future[tuple[_Fetched | _Failed, float]]] = set()
        while True:
            for track in queue:
                inflight.add(pool.submit(_fetch, dz, track, model.sample_rate))
                if len(inflight) >= workers * 2:
                    break
            if not inflight:
                break
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                result, took = fut.result()
                stats.fetch_s += took
                if isinstance(result, _Failed):
                    cache.put_failure(result.track.cache_key, model_id, result.status, result.detail)
                    stats.statuses[result.status] += 1
                else:
                    batch.append(result)
            if len(batch) >= batch_size:
                flush()
    flush()
    cache.commit()
    stats.api_calls = dz.api_calls
    stats.wall_s = time.perf_counter() - t0
    return stats


@dataclass(frozen=True)
class Result:
    """One input line's outcome. `key` echoes the caller's key (falls back to the ISRC)."""

    key: str
    status: Status | Literal["pending"]  # pending = never processed for this model
    vector: np.ndarray | None
    tags: Tags | None


def lookup(tracks: Iterable[Track], cache: Cache, model_id: str) -> list[Result]:
    """Cached results for each input track, in input order (duplicates included)."""
    tracks = list(tracks)
    rows = cache.get_many((t.cache_key for t in tracks), model_id)
    out = []
    for t in tracks:
        row = rows.get(t.cache_key)
        key = t.key or t.cache_key
        out.append(Result(key, row.status, row.vector, row.tags) if row else Result(key, "pending", None, None))
    return out
