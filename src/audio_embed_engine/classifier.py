"""Per-playlist linear classifiers over track vectors (ridge regression, one direction per playlist).

`train` scores every track for every playlist. A labelled track's score comes from models that never
saw it or any song sharing one of its artists, so it is a fair estimate of how the track would score
as a new song. Unlabelled tracks are scored by a model fit on all labelled tracks.

Each playlist also gets a fitted curve from score to chance of belonging, using its held-out scores. A song
gets at most one auto-pick: its most likely playlist, if that chance reaches `confidence` and the playlist's
held-out AUC reaches `min_trust`, so a playlist sound can't separate is never picked from sound alone. The
defaults favour reliable picks over many picks; see bench/REPORT.md for how they were tuned.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RIDGE_LAMBDA = 3.0
CONFIDENCE = 0.9
MIN_TRUST = 0.9  # lowest held-out playlist AUC that may have a cutoff
MIN_POSITIVES = 5  # too few members to fit a confidence curve below this


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Chance a random member outscores a random non-member (ties count half); nan if a class is empty."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_s = scores[order]
    start = 0
    for end in np.flatnonzero(np.r_[sorted_s[1:] != sorted_s[:-1], True]) + 1:
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_ridge(x_test: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Scores (tracks x playlists) from ridge regression. Solves in song space (the dual), so one
    matrix solve covers every playlist, and its size follows the song count rather than the 1280 dims."""
    mean_x, mean_y = x_train.mean(0), y_train.mean(0)
    a = x_train - mean_x
    alpha = np.linalg.solve(a @ a.T + lam * np.eye(len(a)), y_train - mean_y)
    return (x_test - mean_x) @ (a.T @ alpha) + mean_y


def clean_train(train: np.ndarray, test: np.ndarray, artists: Sequence[frozenset[str]]) -> np.ndarray:
    """Training songs sharing no artist with any test song."""
    held = set().union(*(artists[int(i)] for i in test))
    return np.array([i for i in train if not artists[int(i)] & held], dtype=int)


def folds_by_artist(idx: np.ndarray, primary: Sequence[str], n: int, seed: int) -> list[np.ndarray]:
    names = sorted({primary[int(i)] for i in idx})
    np.random.default_rng(seed).shuffle(names)
    fold_of = {a: j % n for j, a in enumerate(names)}
    return [np.array([i for i in idx if fold_of[primary[int(i)]] == f], dtype=int) for f in range(n)]


def out_of_fold(
    x: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    artists: Sequence[frozenset[str]],
    primary: Sequence[str],
    *,
    folds: int = 5,
    repeats: int = 3,
    seed: int = 0,
    lam: float = RIDGE_LAMBDA,
) -> np.ndarray:
    """Held-out scores for the rows in `idx` (len(idx) x playlists), averaged over repeated artist-folds."""
    pos = {int(i): j for j, i in enumerate(idx)}
    total = np.zeros((len(idx), y.shape[1]))
    for r in range(repeats):
        for test in folds_by_artist(idx, primary, folds, seed + r):
            train = clean_train(np.setdiff1d(idx, test), test, artists)
            total[[pos[int(i)] for i in test]] += score_ridge(x[test], x[train], y[train].astype(float), lam)
    return total / repeats


def platt(scores: np.ndarray, pos: np.ndarray, iters: int = 50, l2: float = 1e-3) -> tuple[float, float]:
    """Fit P(member | s) = sigmoid(a*s + b) by Newton with step halving (plain Newton diverges on rare classes)."""
    mu, sd = float(scores.mean()), float(scores.std()) or 1.0
    z, y = (scores - mu) / sd, pos.astype(float)
    n = len(z)

    def loss(a: float, b: float) -> float:
        return float(np.logaddexp(0, -(2 * y - 1) * (a * z + b)).mean() + l2 * a * a / 2)

    a, b = 0.0, float(np.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a * z + b, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-9)
        grad = np.array([(z * (p - y)).sum() / n + l2 * a, (p - y).sum() / n])
        hess = np.array([[(w * z * z).sum() / n + l2, (w * z).sum() / n], [(w * z).sum() / n, w.sum() / n + 1e-9]])
        da, db = np.linalg.solve(hess, grad)
        step, before = 1.0, loss(a, b)
        while step > 1e-4 and loss(a - step * da, b - step * db) > before:
            step /= 2
        a, b = a - step * da, b - step * db
        if max(abs(step * da), abs(step * db)) < 1e-6:
            break
    return a / sd, b - a * mu / sd


@dataclass(frozen=True)
class PlaylistFit:
    auc: float
    # the score at which this playlist can be auto-picked; None if it never can
    cutoff: float | None


def train(
    x: np.ndarray,
    y: np.ndarray,
    artists: Sequence[frozenset[str]],
    primary: Sequence[str],
    *,
    confidence: float = CONFIDENCE,
    min_trust: float = MIN_TRUST,
    lam: float = RIDGE_LAMBDA,
) -> tuple[np.ndarray, list[int | None], list[PlaylistFit]]:
    """Returns scores (every row of x, every playlist), each row's auto-pick column or None, and the fits."""
    labelled = np.flatnonzero(y.any(1))
    oof = out_of_fold(x, y, labelled, artists, primary, lam=lam)
    scores = score_ridge(x, x[labelled], y[labelled].astype(float), lam)
    scores[labelled] = oof

    fits: list[PlaylistFit] = []
    chance = np.zeros(scores.shape)  # stays 0 for playlists that may not be auto-picked
    logit = float(np.log(confidence / (1 - confidence)))
    for p in range(y.shape[1]):
        s, t = oof[:, p], y[labelled, p]
        trust = auc(s, t)
        a, b = platt(s, t) if t.sum() >= MIN_POSITIVES else (0.0, 0.0)
        usable = not np.isnan(trust) and trust >= min_trust and a > 0
        fits.append(PlaylistFit(trust, (logit - b) / a if usable else None))
        if usable:
            chance[:, p] = 1 / (1 + np.exp(-np.clip(a * scores[:, p] + b, -30, 30)))

    best = chance.argmax(1)
    picks = [int(c) if chance[i, c] >= confidence else None for i, c in enumerate(best)]
    return scores, picks, fits
