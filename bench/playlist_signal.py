"""How well does sound predict each playlist, and does trusting it per playlist help?

Usage: python bench/playlist_signal.py <sorter_data_dir> [--out report.json] [--min-size N]

<sorter_data_dir> is PlaylistSorter/data (library.json, vectors.jsonl, hidden.json, rules.json).
Candidates are the playlists the sorter can suggest from sound: not the all-songs playlist, not
ignored, not hidden, no artist rule, at least --min-size embedded songs.

Everything simulates a song by a never-sorted artist. Songs are split into folds by primary artist,
and training drops any song sharing ANY artist with a test song. "Not in this playlist" is only
weak evidence that a song doesn't belong, so absolute precision is a lower bound.

Weights and calibrators are learned from an inner split of the training songs only, then applied to
the held-out fold, so the comparison between scorers is not contaminated by its own test data.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ALL_SONGS = "1xVoRBtLYULxvva0k1OwMn"  # src/core/config.ts in the sorter
IGNORED = {"3fKpVGfURs7G9L3WxrDhtr"}
VOTE_K, TOP_K, RIDGE_LAMBDA = 15, 5, 3.0


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC with tied ranks handled; nan when a class is empty."""
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_s = scores[order]
    # average the ranks of tied scores (kNN votes tie a lot)
    start = 0
    for end in np.flatnonzero(np.r_[sorted_s[1:] != sorted_s[:-1], True]) + 1:
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores: np.ndarray, pos: np.ndarray) -> float:
    if not pos.any():
        return float("nan")
    hits = pos[scores.argsort()[::-1]]
    return float((np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)).mean())


def platt(scores: np.ndarray, pos: np.ndarray, iters: int = 50, l2: float = 1e-3) -> tuple[float, float]:
    """Fit P(pos|s) = sigmoid(a*s + b) by Newton with step halving (plain Newton diverges on rare classes)."""
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


def score_knn(sims: np.ndarray, members: list[np.ndarray], k: int) -> np.ndarray:
    """Mean similarity to each playlist's k most similar training members (queries x playlists)."""
    out = np.full((len(sims), len(members)), -np.inf)
    for p, m in enumerate(members):
        if len(m):
            kk = min(k, len(m))
            out[:, p] = np.partition(sims[:, m], -kk, axis=1)[:, -kk:].mean(1)
    return out


def score_centroid(x_test: np.ndarray, x_train: np.ndarray, members: list[np.ndarray]) -> np.ndarray:
    out = np.full((len(x_test), len(members)), -np.inf)
    for p, m in enumerate(members):
        if len(m):
            c = x_train[m].mean(0)
            out[:, p] = x_test @ (c / np.linalg.norm(c))
    return out


def score_vote(sims: np.ndarray, y_train: np.ndarray, k: int = VOTE_K) -> np.ndarray:
    """The sorter's audio signal: similarity-weighted playlist share among the k nearest sorted songs."""
    top = np.argpartition(sims, -k, axis=1)[:, -k:]
    w = np.maximum(np.take_along_axis(sims, top, 1), 0)
    return (w[:, :, None] * y_train[top]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)


def score_ridge(x_test: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """One linear classifier per playlist in closed form (dual ridge: one solve covers all playlists)."""
    mean_x, mean_y = x_train.mean(0), y_train.mean(0)
    a = x_train - mean_x
    alpha = np.linalg.solve(a @ a.T + lam * np.eye(len(a)), y_train - mean_y)
    return (x_test - mean_x) @ (a.T @ alpha) + mean_y


def clean_train(train: np.ndarray, test: np.ndarray, artists: list[frozenset[str]]) -> np.ndarray:
    """Training songs sharing no artist with any test song."""
    held = set().union(*(artists[i] for i in test))
    return np.array([i for i in train if not artists[i] & held], dtype=int)


def folds_by_artist(idx: np.ndarray, primary: list[str], n: int, seed: int) -> list[np.ndarray]:
    names = sorted({primary[i] for i in idx})
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    fold_of = {a: j % n for j, a in enumerate(names)}
    return [np.array([i for i in idx if fold_of[primary[i]] == f], dtype=int) for f in range(n)]


def all_scores(x, y, test, train) -> dict[str, np.ndarray]:
    members = [train[y[train, p]] for p in range(y.shape[1])]
    xt, xr = x[test], x[train]
    sims = xt @ xr.T
    pos_in_train = [np.searchsorted(train, m) for m in members]
    return {
        "top5": score_knn(sims, pos_in_train, TOP_K),
        "centroid": score_centroid(xt, xr, pos_in_train),
        "vote": score_vote(sims, y[train].astype(float)),
        "ridge": score_ridge(xt, xr, y[train].astype(float)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-labels", action="store_true", help="control: every AUC should land near 0.5")
    args = ap.parse_args()

    lib = json.load(open(args.data / "library.json"))
    hidden = set(json.load(open(args.data / "hidden.json")))
    rules = {k for k, v in json.load(open(args.data / "rules.json")).items() if v}
    vectors = {}
    for line in open(args.data / "vectors.jsonl"):
        d = json.loads(line)
        if d.get("vector"):
            vectors[d["key"]] = d["vector"]
    keys = sorted(vectors)
    pos = {k: i for i, k in enumerate(keys)}

    cands = []
    for p in lib["playlists"]:
        members = sorted({pos[t] for t in p["trackIds"] if t in pos})
        if p["id"] in hidden or p["id"] in rules or p["id"] in IGNORED or p["id"] == ALL_SONGS:
            continue
        if len(members) >= args.min_size:
            cands.append((p["name"], members))
    sorted_tracks = sorted({i for _, m in cands for i in m})
    x = np.stack([np.asarray(vectors[keys[i]], np.float32) for i in sorted_tracks])
    remap = {old: new for new, old in enumerate(sorted_tracks)}
    y = np.zeros((len(x), len(cands)), bool)
    for p, (_, m) in enumerate(cands):
        y[[remap[i] for i in m], p] = True
    if args.shuffle_labels:
        y = y[np.random.default_rng(args.seed).permutation(len(y))]
    tracks = [lib["tracks"][keys[i]] for i in sorted_tracks]
    artists = [frozenset(a["id"] for a in t["artists"]) for t in tracks]
    primary = [t["artists"][0]["id"] for t in tracks]
    print(f"{len(x)} songs, {len(cands)} playlists", flush=True)

    n = len(x)
    methods = ["top5", "centroid", "vote", "ridge"]
    oof = {m: np.full((n, len(cands)), np.nan) for m in methods}
    oof_vote_weighted, oof_ridge_cal = np.full((n, len(cands)), np.nan), np.full((n, len(cands)), np.nan)
    has_train = np.zeros((n, len(cands)), bool)
    trust = np.full((n, len(cands)), np.nan)  # inner-CV AUC of the playlist, as known when scoring each song

    for test in folds_by_artist(np.arange(n), primary, 5, args.seed):
        train = clean_train(np.setdiff1d(np.arange(n), test), test, artists)
        outer = all_scores(x, y, test, train)
        for m in methods:
            oof[m][test] = outer[m]
        has_train[test] = y[train].any(0)

        # inner out-of-fold scores on the training songs: weights and calibrators come only from these
        inner_v, inner_r = np.full((n, len(cands)), np.nan), np.full((n, len(cands)), np.nan)
        for inner_test in folds_by_artist(train, primary, 3, args.seed + 1):
            inner_train = clean_train(np.setdiff1d(train, inner_test), inner_test, artists)
            s = all_scores(x, y, inner_test, inner_train)
            inner_v[inner_test], inner_r[inner_test] = s["vote"], s["ridge"]
        for p in range(len(cands)):
            seen = train[~np.isnan(inner_v[train, p])]
            a = auc(inner_v[seen, p], y[seen, p]) if len(seen) else float("nan")
            weight = 0.0 if np.isnan(a) else max(0.0, (a - 0.5) * 2)
            oof_vote_weighted[test, p] = outer["vote"][:, p] * weight
            trust[test, p] = auc(inner_r[seen, p], y[seen, p]) if len(seen) else float("nan")
            if y[seen, p].sum() >= 5:
                pa, pb = platt(inner_r[seen, p], y[seen, p])
                oof_ridge_cal[test, p] = 1 / (1 + np.exp(-np.clip(pa * outer["ridge"][:, p] + pb, -30, 30)))
            else:
                oof_ridge_cal[test, p] = y[train, p].mean()

    oof["vote x playlist weight"], oof["ridge calibrated"] = oof_vote_weighted, oof_ridge_cal
    rng = np.random.default_rng(args.seed)

    # ranking: for each song, are its true playlists in the top 3? Only playlists with training data count.
    overall = {}
    per_query = {}
    for name, s in oof.items():
        recall, hit = [], []
        for i in range(n):
            truth = np.flatnonzero(y[i] & has_train[i])
            if not len(truth):
                continue
            ok = np.where(has_train[i], s[i], -np.inf)
            top = np.argsort(-ok)[:3]
            got = len(set(top) & set(truth))
            recall.append(got / len(truth))
            hit.append(float(got > 0))
        per_query[name] = np.array(recall)
        overall[name] = {"recall@3": round(float(np.mean(recall)), 4), "hit@3": round(float(np.mean(hit)), 4)}
    pop = y.sum(0)
    recall = []
    for i in range(n):
        truth = np.flatnonzero(y[i] & has_train[i])
        if len(truth):
            ok = np.where(has_train[i], pop, -1)
            recall.append(len(set(np.argsort(-ok)[:3]) & set(truth)) / len(truth))
    overall["biggest 3 playlists"] = {"recall@3": round(float(np.mean(recall)), 4)}
    for name in ("vote x playlist weight", "ridge", "ridge calibrated"):
        d = per_query[name] - per_query["vote"]
        boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(1000)]
        overall[name]["delta_vs_vote"] = [round(float(d.mean()), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]

    # would a confidence cutoff be trustworthy? share of confident top picks that are real memberships
    cal = oof["ridge calibrated"]
    confident = {}
    for label, low, high in (("trust >= 0.90", 0.9, 1.01), ("trust 0.75-0.90", 0.75, 0.9), ("trust < 0.75", 0.0, 0.75)):
        tier = (trust >= low) & (trust < high) & has_train
        confident[label] = {}
        for cut in (0.3, 0.5, 0.7, 0.9):
            picks = (cal >= cut) & tier
            if picks.any():
                confident[label][str(cut)] = {"picks": int(picks.sum()), "real": round(float(y[picks].mean()), 3)}

    table = []
    for p, (name, _) in enumerate(cands):
        mask = has_train[:, p]
        n_pos = int((y[:, p] & mask).sum())
        row = {"playlist": name, "songs": int(y[:, p].sum()), "testable": n_pos, "artists": len({primary[i] for i in np.flatnonzero(y[:, p])})}
        if n_pos < 5:
            table.append(row | {"auc": None, "note": "too few songs from other artists to test"})
            continue
        for m in ("vote", "top5", "centroid", "ridge"):
            row[f"auc_{m}"] = round(auc(oof[m][mask, p], y[mask, p]), 3)
        s, t = oof["ridge"][mask, p], y[mask, p]
        boots = []
        idx_pos, idx_neg = np.flatnonzero(t), np.flatnonzero(~t)
        for _ in range(200):
            i = np.r_[rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))]
            boots.append(auc(s[i], t[i]))
        base = float(t.mean())
        row |= {
            "auc": row["auc_ridge"],
            "auc_lo": round(float(np.percentile(boots, 2.5)), 3),
            "auc_hi": round(float(np.percentile(boots, 97.5)), 3),
            "base_rate": round(base, 3),
            "ap_lift": round(average_precision(s, t) / base, 2),
        }
        table.append(row)
    table.sort(key=lambda r: -(r["auc"] or -1))

    print(json.dumps(overall, indent=1))
    print(json.dumps(confident, indent=1))
    print(f"{'auc':>5} {'95% CI':>13} {'lift':>5} {'songs':>5} {'artists':>7}  playlist")
    for r in table:
        if r["auc"] is None:
            print(f"{'-':>5} {'':>13} {'':>5} {r['songs']:>5} {r['artists']:>7}  {r['playlist']}  ({r['note']})")
        else:
            print(f"{r['auc']:>5.3f} [{r['auc_lo']:.3f},{r['auc_hi']:.3f}] {r['ap_lift']:>5.1f} {r['songs']:>5} {r['artists']:>7}  {r['playlist']}")
    if args.out:
        args.out.write_text(json.dumps({"overall": overall, "confident": confident, "playlists": table}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
