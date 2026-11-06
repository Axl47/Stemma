"""Quality metrics for embeddings (rows L2-normalised, so dot product = cosine similarity).

- pair_auc: P(sim of a random same-playlist pair > sim of a random different-playlist pair).
  0.5 = chance. With exclude_shared_artist, pairs sharing any artist are dropped first.
- playlist_ranking: leave-one-out version of the sorter's task. For each track, rank playlists
  by similarity to their *other* members, ignoring every member that shares an artist with it
  (simulates "artist I've never sorted"). Reports recall@3 / hit@3 for two playlist scorers:
  centroid (cosine to the mean member) and top3 (mean of the 3 most similar members).
- style_knn: on the hand-built fixture, 5-NN style vote and same-style pair AUC, again
  excluding same-artist neighbours.
"""

from collections import Counter

import numpy as np


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks."""
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    ranks[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pair_auc(
    keys: list[str],
    emb: np.ndarray,
    labels: dict[str, set[str]],
    artists: dict[str, set[str]],
    exclude_shared_artist: bool,
) -> float:
    sims = emb @ emb.T
    pos, neg = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if exclude_shared_artist and artists[keys[i]] & artists[keys[j]]:
                continue
            (pos if labels[keys[i]] & labels[keys[j]] else neg).append(sims[i, j])
    return _auc(np.array(pos), np.array(neg))


def playlist_ranking(
    keys: list[str], emb: np.ndarray, labels: dict[str, set[str]], artists: dict[str, set[str]], k: int = 3
) -> dict[str, float]:
    sims = emb @ emb.T
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        for p in labels[key]:
            members.setdefault(p, []).append(i)
    stats: dict[str, list[float]] = {f"{s}_{m}@{k}": [] for s in ("centroid", "top3") for m in ("recall", "hit")}
    stats["chance_recall@3"] = []
    for q, qkey in enumerate(keys):
        eligible = {p: [i for i in ms if i != q and not artists[keys[i]] & artists[qkey]] for p, ms in members.items()}
        eligible = {p: ms for p, ms in eligible.items() if ms}
        truth = labels[qkey] & eligible.keys()
        if not truth:
            continue
        names = list(eligible)
        centroid = [float(emb[q] @ (c := emb[ms].mean(0)) / np.linalg.norm(c)) for ms in eligible.values()]
        top3 = [float(np.sort(sims[q, ms])[-3:].mean()) for ms in eligible.values()]
        for scorer, scores in (("centroid", centroid), ("top3", top3)):
            top = {names[i] for i in np.argsort(scores)[-k:]}
            stats[f"{scorer}_recall@{k}"].append(len(top & truth) / len(truth))
            stats[f"{scorer}_hit@{k}"].append(float(bool(top & truth)))
        stats["chance_recall@3"].append(min(k / len(names), 1.0))
    out = {name: float(np.mean(v)) for name, v in stats.items()}
    out["ranking_queries"] = float(len(stats["chance_recall@3"]))
    return out


def style_knn(
    keys: list[str], emb: np.ndarray, style: dict[str, str], artist: dict[str, str], k: int = 5
) -> dict[str, float]:
    sims = emb @ emb.T
    correct, pos, neg = 0, [], []
    for q, qk in enumerate(keys):
        others = [i for i in range(len(keys)) if i != q and artist[keys[i]] != artist[qk]]
        nn = sorted(others, key=lambda i: -sims[q, i])[:k]
        vote = Counter(style[keys[i]] for i in nn).most_common(1)[0][0]
        correct += vote == style[qk]
        for i in others:
            if i > q:
                (pos if style[keys[i]] == style[qk] else neg).append(sims[q, i])
    return {"fixture_style_knn5_acc": correct / len(keys), "fixture_style_pair_auc": _auc(np.array(pos), np.array(neg))}
