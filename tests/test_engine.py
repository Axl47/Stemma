"""Engine + cache behaviour with a fake Deezer and model (no network, no weights)."""

import time

import numpy as np
import pytest

from audio_embed_engine import Cache, Embedding, Track, embed_tracks, lookup
from audio_embed_engine.deezer import Deezer, Preview, Resolution


class FakeDeezer(Deezer):
    def __init__(self, missing: set[str]) -> None:
        super().__init__()
        self.missing = missing
        self.resolved: list[str] = []

    def resolve(self, track: Track) -> Resolution:
        self.resolved.append(track.cache_key)
        return "not_found" if track.cache_key in self.missing else Preview(1, track.cache_key)

    def download(self, preview: Preview) -> bytes:
        return preview.url.encode()  # lets fake decoders know which track they got


class FakeModel:
    id = "fake/v1"
    sample_rate = 16_000

    def embed(self, audios: list[np.ndarray]) -> list[Embedding]:
        return [Embedding(np.array([0.6, 0.8], np.float32), {"styles": {"x": 1.0}}) for _ in audios]


@pytest.fixture(autouse=True)
def fake_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audio_embed_engine.engine.decode", lambda data, rate: np.zeros(rate, np.float32))


def track(isrc: str | None, key: str | None = None) -> Track:
    return Track(isrc=isrc, title="t", artists=("a",), duration_ms=1000, key=key)


def test_second_run_only_processes_new_tracks(tmp_path) -> None:
    cache = Cache(tmp_path / "c.sqlite")
    first = [track("AAA"), track("BBB"), track("MISSING")]
    dz = FakeDeezer(missing={"MISSING"})
    stats = embed_tracks(first, cache, FakeModel(), deezer=dz)
    assert stats.statuses == {"ok": 2, "not_found": 1}

    dz.resolved.clear()
    stats = embed_tracks(first + [track("CCC")], cache, FakeModel(), deezer=dz)
    assert dz.resolved == ["CCC"]  # cached ok rows and the cached failure are both skipped
    assert stats.cached == 3


def test_failures_retry_after_expiry_and_new_model_adds_rows(tmp_path) -> None:
    cache = Cache(tmp_path / "c.sqlite")
    embed_tracks([track("MISSING")], cache, FakeModel(), deezer=FakeDeezer({"MISSING"}))
    assert cache.pending(["MISSING"], "fake/v1") == set()
    assert cache.pending(["MISSING"], "fake/v1", retry_failed=True) == {"MISSING"}
    cache.db.execute("UPDATE embeddings SET retry_after = ?", (time.time() - 1,))
    assert cache.pending(["MISSING"], "fake/v1") == {"MISSING"}
    assert cache.pending(["MISSING"], "other-model/v1") == {"MISSING"}


def test_lookup_echoes_caller_key_and_reports_pending(tmp_path) -> None:
    cache = Cache(tmp_path / "c.sqlite")
    tracks = [track("aaa", key="spotify1"), track(None, key="spotify2"), track("AAA", key="spotify3")]
    embed_tracks(tracks[:2], cache, FakeModel(), deezer=FakeDeezer(set()))
    results = lookup(tracks + [track("ZZZ", key="spotify4")], cache, "fake/v1")
    assert [(r.key, r.status) for r in results] == [
        ("spotify1", "ok"),
        ("spotify2", "ok"),
        ("spotify3", "ok"),
        ("spotify4", "pending"),
    ]
    assert results[0].vector is not None and results[0].vector.tolist() == pytest.approx([0.6, 0.8])


def test_one_bad_track_does_not_sink_the_batch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Picky(FakeModel):
        def embed(self, audios: list[np.ndarray]) -> list[Embedding]:
            if any(len(a) == 1 for a in audios):
                raise ValueError("bad audio")
            return super().embed(audios)

    sizes = {"GOOD1": 16_000, "BAD": 1, "GOOD2": 16_000}
    monkeypatch.setattr(
        "audio_embed_engine.engine.decode", lambda data, rate: np.zeros(sizes[data.decode()], np.float32)
    )
    cache = Cache(tmp_path / "c.sqlite")
    stats = embed_tracks([track(k) for k in sizes], cache, Picky(), deezer=FakeDeezer(set()))
    assert stats.statuses == {"ok": 2, "error": 1}
    assert cache.get_many(["BAD"], "fake/v1")["BAD"].status == "error"
