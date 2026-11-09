"""Tune the playlist classifier: its penalty, and when a score is good enough to auto-check a playlist.

Usage: python bench/classifier_tuning.py <sorter_data_dir> [--seeds N]

Same songs, playlists and never-seen-artist folds as playlist_signal.py. Every cutoff is chosen from held-out
scores inside the training songs, exactly as `audio_embed_engine.classifier.train` does for the app, then
judged on a separate test fold. "Real" is a lower bound: a song missing from a playlist may still fit it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audio_embed_engine.classifier import auc, clean_train, folds_by_artist, out_of_fold, platt, score_ridge

ALL_SONGS, IGNORED = "1xVoRBtLYULxvva0k1OwMn", {"3fKpVGfURs7G9L3WxrDhtr"}
CONFIDENCES = (0.5, 0.6, 0.7, 0.8, 0.9)
TRUSTS = (0.0, 0.85, 0.9, 0.93)


def load(data: Path):
    lib = json.load(open(data / "library.json"))
    hidden = set(json.load(open(data / "hidden.json")))
    rules = {k for k, v in json.load(open(data / "rules.json")).items() if v}
    vec = {}
    for line in open(data / "vectors.jsonl"):
        d = json.loads(line)
        if d.get("vector"):
            vec[d["key"]] = d["vector"]
    cands = []
    for p in lib["playlists"]:
        members = [t for t in dict.fromkeys(p["trackIds"]) if t in vec]
        if p["id"] not in hidden | rules | IGNORED and p["id"] != ALL_SONGS and len(members) >= 12:
            cands.append((p["name"], members))
    tracks = sorted({t for _, m in cands for t in m})
    pos = {t: i for i, t in enumerate(tracks)}
    x = np.array([vec[t] for t in tracks], np.float32)
    y = np.zeros((len(x), len(cands)), bool)
    for p, (_, m) in enumerate(cands):
        y[[pos[t] for t in m], p] = True
    artists = [frozenset(a["id"] for a in lib["tracks"][t]["artists"]) for t in tracks]
    primary = [lib["tracks"][t]["artists"][0]["id"] for t in tracks]
    return x, y, artists, primary, [n for n, _ in cands]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    x, y, artists, primary, _names = load(a.data)
    n, idx = len(x), np.arange(len(x))
    print(f"{n} songs, {y.shape[1]} playlists", flush=True)

    # (confidence, min trust, top pick only) -> picks, right picks, songs with a right pick
    keys = [(c, t, top) for c in CONFIDENCES for t in TRUSTS for top in (False, True)]
    tally = {k: [0, 0, 0] for k in keys}
    fold_precision: dict[tuple, list[float]] = {k: [] for k in keys}
    active: dict[tuple, list[int]] = {k: [] for k in keys}
    tests = 0
    for seed in range(a.seeds):
        for test in folds_by_artist(idx, primary, 5, seed + 100):
            train = clean_train(np.setdiff1d(idx, test), test, artists)
            scores = score_ridge(x[test], x[train], y[train].astype(float))
            oof = out_of_fold(x, y, train, artists, primary, folds=5, repeats=2, seed=seed + 200)
            fits = []
            for p in range(y.shape[1]):
                s_, t_ = oof[:, p], y[train, p]
                fits.append((auc(s_, t_), *(platt(s_, t_) if t_.sum() >= 5 else (0.0, -30.0))))
            tests += len(test)
            prob = np.column_stack([1 / (1 + np.exp(-np.clip(pa * scores[:, p] + pb, -30, 30))) for p, (_, pa, pb) in enumerate(fits)])
            for k in keys:
                conf, trust, top = k
                usable = np.array([not np.isnan(u) and u >= trust and pa > 0 for u, pa, _ in fits])
                hits = (prob >= conf) & usable
                if top:
                    best = (prob * usable).argmax(1)
                    only = np.zeros(hits.shape, bool)
                    only[np.arange(len(hits)), best] = True
                    hits &= only
                real = hits & y[test]
                tally[k][0] += int(hits.sum())
                tally[k][1] += int(real.sum())
                tally[k][2] += int(real.any(1).sum())
                active[k].append(int(usable.sum()))
                if hits.any():
                    fold_precision[k].append(float(real.sum() / hits.sum()))
    print(f"\nauto-check policies ({tests} tests):")
    print(f"{'conf':>4} {'minAUC':>6} {'top only':>8} {'lists':>6} {'checks/song':>11} {'real':>6} {'fold range':>11} {'right/song':>10}")
    for k, (picks, real, songs) in tally.items():
        fp = fold_precision[k]
        rng = f"{min(fp):.2f}-{max(fp):.2f}" if fp else "-"
        print(f"{k[0]:>4} {k[1]:>6} {k[2]!s:>8} {np.mean(active[k]):>6.1f} {picks / tests:>11.3f} {real / max(picks, 1):>6.3f} {rng:>11} {songs / tests:>10.3f}")


if __name__ == "__main__":
    main()
