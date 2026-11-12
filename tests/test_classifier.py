import json
import sys
from pathlib import Path

import numpy as np
import pytest

from audio_embed_engine import cli
from audio_embed_engine.classifier import auc, score_ridge, train, with_block, with_language
from audio_embed_engine.language import detect_language


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_language_separates_songs_that_sound_alike() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = unit(rng.normal(size=(n, 32)))
    languages = ["es" if i % 2 else "en" for i in range(n)]
    y = np.array([[i % 2 == 1] for i in range(n)])

    joined, names = with_language(x, languages, np.ones(n, bool), min_songs=15)
    held_out = np.arange(0, n, 5)
    train_rows = np.setdiff1d(np.arange(n), held_out)
    plain = score_ridge(x[held_out], x[train_rows], y[train_rows].astype(float))
    enriched = score_ridge(joined[held_out], joined[train_rows], y[train_rows].astype(float))

    assert names == ["en", "es"]
    assert auc(enriched[:, 0], y[held_out, 0]) > 0.95
    assert auc(plain[:, 0], y[held_out, 0]) < 0.7


def test_rare_or_unknown_language_leaves_a_song_on_sound_alone() -> None:
    x = unit(np.random.default_rng(1).normal(size=(4, 8)))

    joined, names = with_language(x, ["en", "en", "sw", None], np.ones(4, bool), min_songs=2)

    assert names == ["en"]
    assert np.allclose(joined[2:, :8], x[2:]) and np.allclose(joined[2:, 8:], 0)
    assert np.allclose(np.linalg.norm(joined, axis=1), 1)


def test_detect_language_reads_scripts_and_declines_short_text() -> None:
    english = "I walk alone along the river every night and think about the words you said to me " * 3
    assert detect_language(english) == "en"
    assert detect_language(english + "あなたのことを") == "ja"
    assert detect_language("too short") is None


def test_cutoff_only_for_a_playlist_sound_can_separate() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 32))
    # playlist 0 follows the first dimension; playlist 1 is random, so sound can't predict it
    x[:200, 0] += 6
    # playlist 2 labels the rest, so every song is in the training universe
    y = np.zeros((n, 3), bool)
    y[:200, 0] = True
    y[rng.choice(n, 200, replace=False), 1] = True
    y[200:, 2] = True
    artists = [frozenset({f"a{i // 2}"}) for i in range(n)]
    primary = [f"a{i // 2}" for i in range(n)]

    scores, picks, fits = train(unit(x), y, artists, primary, confidence=0.7)

    assert fits[0].auc > 0.95 and fits[0].cutoff is not None
    assert fits[1].auc < 0.7 and fits[1].cutoff is None
    assert auc(scores[:, 0], y[:, 0]) > 0.95
    assert 1 not in picks and picks[:200].count(0) > 100


def test_a_labelled_song_is_scored_without_seeing_its_own_label() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = unit(rng.normal(size=(n, 600)))  # more dims than songs: a model that saw the label memorizes it
    y = np.zeros((n, 1), bool)
    y[:150, 0] = True
    artists = [frozenset({f"a{i}"}) for i in range(n)]

    scores, _, _ = train(x, y, artists, [f"a{i}" for i in range(n)])

    assert abs(auc(scores[:, 0], y[:, 0]) - 0.5) < 0.15


def test_a_song_without_lyrics_gets_the_fit_songs_mean_and_is_scored_from_sound() -> None:
    rng = np.random.default_rng(2)
    x = unit(rng.normal(size=(6, 8)))
    block = unit(rng.normal(size=(6, 4)))
    present = np.array([True, True, True, False, True, False])
    fit = np.array([True, True, False, True, True, False])  # row 2 has lyrics but isn't a fit song

    joined = with_block(x, block, present, fit, weight=2.0)

    mean = (block[0] + block[1] + block[4]) / 3
    assert joined.shape == (6, 12) and np.allclose(joined[:, :8], x)
    assert np.allclose(joined[[0, 1, 2, 4], 8:], 2.0 * block[[0, 1, 2, 4]])
    assert np.allclose(joined[[3, 5], 8:], 2.0 * mean)


def test_classify_without_lyrics_is_the_sound_and_language_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(3)
    n = 120
    x = unit(rng.normal(size=(n, 16)))
    x[: n // 2, 0] += 0.5
    songs = [
        {"key": f"s{i}", "artists": [f"a{i}"], "labels": ["p"] if i < n // 2 else ["q"], "language": "en"}
        for i in range(n)
    ]
    labels, vectors, out = tmp_path / "labels.jsonl", tmp_path / "vectors.jsonl", tmp_path / "out.json"
    labels.write_text("\n".join(json.dumps(s) for s in songs))
    vectors.write_text("\n".join(json.dumps({"key": f"s{i}", "vector": x[i].tolist()}) for i in range(n)))
    monkeypatch.setattr(
        sys, "argv", ["audio-embed", "classify", str(labels), "--vectors", str(vectors), "--out", str(out)]
    )

    cli.main()

    y = np.array([[i < n // 2, i >= n // 2] for i in range(n)])
    joined, _ = with_language(x, ["en"] * n, y.any(1), min_songs=15)
    scores, picks, _ = train(joined, y, [frozenset({f"a{i}"}) for i in range(n)], [f"a{i}" for i in range(n)])
    saved = json.loads(out.read_text())
    assert saved["scores"] == np.round(scores, 3).tolist() and saved["picks"] == picks
