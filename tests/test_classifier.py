import numpy as np

from audio_embed_engine.classifier import auc, train


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


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
