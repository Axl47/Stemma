"""Does a small learned transform of the vectors make "similar" match how this listener sorts songs?

Usage (from the Stemma repo; torch comes from the bench group):
    PYTHONPATH=bench uv run --group bench python bench/personal_metric.py <sorter_data_dir> [--out results.json]

<sorter_data_dir> is Cento's data/ (classifier-input.jsonl, vectors.jsonl, library.json untouched).
Its classifier-input.jsonl already holds only the playlists the sorter counts: no all-songs playlist,
no ignored or hidden ones, no artist-rule playlists, no single-album playlists. Nothing here is written
to the repo, and nothing leaves the machine.

Candidates, simplest first:
  base    cosine on the raw vectors
  center  subtract the mean, renormalise
  pcaw    PCA to d dims, whitened to a power alpha (alpha 0 = plain PCA, 1 = full whitening); unsupervised
  ridge   ridge regression to the playlist-membership matrix, rank-reduced; at most one axis per playlist
  blend   pcaw plus the ridge axes, as a weighted mix of the two cosine similarities (reaches 32..256+ dims)
  contra  linear map on the top PCA directions, trained with a supervised-contrastive loss (torch, CPU)
  mlp     the same with a small MLP head (only worth running if contra/ridge clearly beat base)

Evaluation simulates songs by never-sorted artists. Songs are split into folds by primary artist; the
test songs are the fold, and the training songs are every other song sharing NO artist with any test
song (the featuring graph links 55% of the library into one group, so whole connected groups can't be
kept together; dropping every training song that touches a test artist gives the same guarantee).
Nothing (mean, PCA, scaler, regression, network) is ever fitted on a test song. Unsupervised fits may
also use the unlabelled songs of non-test artists; supervised fits use only labelled training songs.

Metrics (all on test songs, similarity = cosine in the transformed space):
  pair AUC     same-playlist vs different-playlist pairs among test songs, shared-artist pairs dropped
               (definition of bench/metrics.py, vectorised)
  recall@3     place a test song: rank each playlist by the mean cosine to its 5 most similar training
               members, take the top 3 (bench/library_eval.py's top5 scorer); share of the song's playlists
               that appear. Also the centroid scorer. Playlists with no training member are not candidates.
  nn15         share of a test song's 15 nearest training songs that share a playlist with it
  small        the same, restricted to songs in a playlist under SMALL songs (see below)
  pop3         share of the top-3 suggestions that are the 3 biggest candidate playlists (collapse check)

Playlist size: pair AUC and nn15 are song-level, so big playlists dominate them; that is why every metric
is also reported for small playlists only. Supervised fits ignore playlists with fewer than MIN_TRAIN
training members, and "balance" options standardise each playlist's target (ridge) or weight anchors by
1/sqrt(size of the song's smallest playlist) (contrastive) so small playlists are not drowned out.
"""

import os

# Shared machine: cap every BLAS / torch pool at 4 threads before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "4"

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from metrics import _auc

from audio_embed_engine.classifier import clean_train, folds_by_artist

SMALL = 40  # playlists with fewer songs than this are "small"
MIN_TRAIN = 5  # supervised fits skip playlists with fewer training members in the fold
VOTE_K, TOP_K, NN_K = 5, 3, 15

Transform = Callable[[np.ndarray], np.ndarray]  # rows in, L2-normalised rows out


def normalise(z: np.ndarray) -> np.ndarray:
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


@dataclass
class Data:
    keys: list[str]
    x: np.ndarray  # (n, 1280) float64, L2-normalised
    artists: list[frozenset[str]]
    primary: list[str]
    y: np.ndarray  # (n, P) bool membership over the counted playlists
    size: np.ndarray  # (P,) songs per playlist
    labelled: np.ndarray  # indices of songs in at least one playlist


def load(data_dir: Path) -> Data:
    vectors = {}
    for line in open(data_dir / "vectors.jsonl"):
        row = json.loads(line)
        if row["status"] == "ok":
            vectors[row["key"]] = row["vector"]
    rows = [json.loads(line) for line in open(data_dir / "classifier-input.jsonl")]
    rows = [r for r in rows if r["key"] in vectors and r["artists"]]
    playlists = sorted({p for r in rows for p in r["labels"]})
    column = {p: i for i, p in enumerate(playlists)}
    y = np.zeros((len(rows), len(playlists)), bool)
    for i, r in enumerate(rows):
        for p in r["labels"]:
            y[i, column[p]] = True
    x = np.array([vectors[r["key"]] for r in rows], dtype=np.float64)
    return Data(
        keys=[r["key"] for r in rows],
        x=normalise(x),
        artists=[frozenset(r["artists"]) for r in rows],
        primary=[r["artists"][0] for r in rows],
        y=y,
        size=y.sum(0),
        labelled=np.flatnonzero(y.any(1)),
    )


class Fold:
    """One artist-grouped split plus everything about it that does not depend on the transform."""

    def __init__(self, d: Data, test: np.ndarray):
        self.d = d
        self.test = test
        everyone = np.arange(len(d.keys))
        self.train = clean_train(np.setdiff1d(d.labelled, test), test, d.artists)
        self.unsup = clean_train(np.setdiff1d(everyone, test), test, d.artists)
        test_artists = set().union(*(d.artists[int(i)] for i in test))
        assert not any(d.artists[int(i)] & test_artists for i in self.train)  # no shared artist, ever
        assert not set(self.train.tolist()) & set(test.tolist()) and not set(self.unsup.tolist()) & set(test.tolist())
        # Eligible playlists for supervised fits: enough training members.
        self.cols = np.flatnonzero(d.y[self.train].sum(0) >= MIN_TRAIN)
        self.small = d.size < SMALL
        y_te, y_tr = d.y[test], d.y[self.train]
        y_small = d.y & self.small
        self.in_small = y_small[test].any(1)
        # Pair structure among test songs (upper triangle).
        self.iu = np.triu_indices(len(test), 1)
        names = sorted({a for i in test for a in d.artists[int(i)]})
        a_index = {a: j for j, a in enumerate(names)}
        inc = np.zeros((len(test), len(names)), np.float32)
        for r, i in enumerate(test):
            for a in d.artists[int(i)]:
                inc[r, a_index[a]] = 1
        self.shared_artist = ((inc @ inc.T) > 0)[self.iu]
        self.share = (y_te.astype(np.float32) @ y_te.T.astype(np.float32) > 0)[self.iu]
        self.share_small = (y_small[test].astype(np.float32) @ y_small[test].T.astype(np.float32) > 0)[self.iu]
        # Test x train sharing, for neighbour quality.
        self.share_tt = y_te.astype(np.float32) @ y_tr.T.astype(np.float32) > 0
        self.share_tt_small = y_small[test].astype(np.float32) @ y_small[self.train].T.astype(np.float32) > 0
        self.chance_nn = float(self.share_tt.mean())
        # Training songs sharing an artist (for the contrastive loss).
        names_tr = sorted({a for i in self.train for a in d.artists[int(i)]})
        a_tr = {a: j for j, a in enumerate(names_tr)}
        inc_tr = np.zeros((len(self.train), len(names_tr)), np.float32)
        for r, i in enumerate(self.train):
            for a in d.artists[int(i)]:
                inc_tr[r, a_tr[a]] = 1
        self.same_artist_tr = (inc_tr @ inc_tr.T) > 0
        self._pca: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def pca(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mean, eigenvalues desc, eigenvectors as columns) of the unsupervised-fit songs."""
        if self._pca is None:
            fit = self.d.x[self.unsup]
            mu = fit.mean(0)
            ev, vec = np.linalg.eigh(np.cov(fit - mu, rowvar=False))
            self._pca = (mu, ev[::-1].clip(1e-12), vec[:, ::-1])
        return self._pca


# ---------------------------------------------------------------------------------------------------------
# Transforms. Each `make_*` fits on the fold's training songs and returns a function over any rows.


def make_base(_f: Fold, _cols: np.ndarray) -> Transform:
    return normalise


def make_center(f: Fold, _cols: np.ndarray) -> Transform:
    mu = f.pca()[0]
    return lambda x: normalise(x - mu)


def make_pcaw(f: Fold, _cols: np.ndarray, *, d: int, alpha: float) -> Transform:
    mu, ev, vec = f.pca()
    w = vec[:, :d] * ev[:d] ** (-alpha / 2)
    return lambda x: normalise((x - mu) @ w)


def make_ridge(f: Fold, cols: np.ndarray, *, lam: float, balance: bool, rank: int | None) -> Transform:
    """Ridge regression of (standardised) playlist membership on the vectors, then the fitted values as
    the embedding, optionally cut to the `rank` strongest directions. lam is relative to the mean eigenvalue."""
    x, y = f.d.x[f.train], f.d.y[f.train][:, cols].astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    t = y - y.mean(0)
    if balance:
        t = t / np.maximum(y.std(0), 1e-9)
    gram = xc.T @ xc
    w = np.linalg.solve(gram + lam * np.trace(gram) / len(gram) * np.eye(len(gram)), xc.T @ t)
    if rank is not None and rank < w.shape[1]:
        _, _, vt = np.linalg.svd(xc @ w, full_matrices=False)
        w = w @ vt[:rank].T
    return lambda z: normalise((z - mu) @ w)


def blend(parts: Sequence[tuple[float, Transform]]) -> Transform:
    """Weights are shares of the similarity: cosine of the result = sum(w * cosine of each part)."""
    return lambda x: np.hstack([np.sqrt(w) * t(x) for w, t in parts])


def make_blend(f: Fold, cols: np.ndarray, *, d: int, alpha: float, beta: float, **ridge) -> Transform:
    return blend([(1 - beta, make_pcaw(f, cols, d=d, alpha=alpha)), (beta, make_ridge(f, cols, **ridge))])


def make_contra(
    f: Fold,
    cols: np.ndarray,
    *,
    d: int,
    steps: int,
    lr: float,
    tau: float = 0.3,
    reg: float = 0.0,
    balance: bool = True,
    alpha: float = 0.0,
    k: int = 256,
    hidden: int = 0,
) -> Transform:
    """A map from the top-k PCA directions to d dims, trained with a supervised-contrastive loss: positives
    share a playlist, negatives share none, same-artist pairs are left out of both (so the map can't learn
    "this is that artist"). Linear (hidden=0) starts from plain truncation (= pcaw) and is pulled back toward
    it by `reg`; with hidden > 0 it is a one-hidden-layer MLP with a linear skip path, starting at truncation.
    Whole-batch training, so steps are full passes. steps, tau and alpha were picked on a dev split (seed 7)."""
    import torch

    torch.set_num_threads(4)
    mu, ev, vec = f.pca()
    basis = vec[:, :k] * ev[:k] ** (-alpha / 2)
    feats = torch.tensor((f.d.x[f.train] - mu) @ basis, dtype=torch.float32)
    y = f.d.y[f.train][:, cols].astype(np.float32)
    share = torch.tensor(y @ y.T > 0)
    same = torch.tensor(f.same_artist_tr)
    pos, allowed = share & ~same, ~same
    n_pos = pos.sum(1)
    if balance:
        smallest = np.where(y > 0, f.d.size[cols][None, :], 10**9).min(1)
        w_song = torch.tensor(1 / np.sqrt(smallest), dtype=torch.float32)
    else:
        w_song = torch.ones(len(feats))
    w_song = w_song * (n_pos > 0)  # anchors need a positive
    w0 = torch.eye(k, d)
    w = w0.clone().requires_grad_(True)
    mlp = (
        torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d)) if hidden else None
    )
    if mlp is not None:
        torch.nn.init.zeros_(mlp[2].weight)  # starts exactly at the linear skip path
        torch.nn.init.zeros_(mlp[2].bias)
    params = [w] + (list(mlp.parameters()) if mlp is not None else [])
    opt = torch.optim.Adam(params, lr=lr)

    def embed(h: torch.Tensor) -> torch.Tensor:
        out = h @ w
        return torch.nn.functional.normalize(out + mlp(h) if mlp is not None else out, dim=1)

    for _ in range(steps):
        z = embed(feats)
        s = (z @ z.T / tau).masked_fill(~allowed, float("-inf"))
        logp = s - torch.logsumexp(s, dim=1, keepdim=True)
        per_song = -(logp.masked_fill(~pos, 0).sum(1)) / n_pos.clamp(min=1)
        loss = (w_song * per_song).sum() / w_song.sum() + reg * ((w - w0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def transform(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return embed(torch.tensor((x - mu) @ basis, dtype=torch.float32)).numpy().astype(np.float64)

    return transform


# ---------------------------------------------------------------------------------------------------------
# Metrics


def pair_auc(sim_pairs: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    if not pos.any() or not neg.any():
        return float("nan")
    return _auc(sim_pairs[pos], sim_pairs[neg])


def evaluate(f: Fold, z_all: np.ndarray) -> dict[str, float]:
    d = f.d
    z_te, z_tr = z_all[f.test], z_all[f.train]
    s_tt = z_te @ z_te.T
    sims = s_tt[f.iu]
    ok = ~f.shared_artist
    small_any = f.in_small[f.iu[0]] | f.in_small[f.iu[1]]
    out = {
        "pair_auc": pair_auc(sims, f.share & ok, ~f.share & ok),
        "pair_auc_small": pair_auc(sims, f.share_small & ok, ~f.share & ok & small_any),
    }
    s = z_te @ z_tr.T  # test x train (no artist is shared by construction)
    top = np.argpartition(s, -NN_K, axis=1)[:, -NN_K:]
    out["nn15"] = float(np.take_along_axis(f.share_tt, top, 1).mean())
    out["nn15_lift"] = float((np.take_along_axis(f.share_tt, top, 1).mean(1) - f.share_tt.mean(1)).mean())
    out["nn15_small"] = float(np.take_along_axis(f.share_tt_small, top, 1)[f.in_small].mean())
    # Placing a song: score every playlist from its training members.
    y_tr = d.y[f.train]
    n_pl = d.y.shape[1]
    knn = np.full((len(f.test), n_pl), -np.inf)
    cen = np.full((len(f.test), n_pl), -np.inf)
    for p in range(n_pl):
        m = np.flatnonzero(y_tr[:, p])
        if len(m):
            kk = min(VOTE_K, len(m))
            knn[:, p] = np.partition(s[:, m], -kk, axis=1)[:, -kk:].mean(1)
            c = z_tr[m].mean(0)
            cen[:, p] = z_te @ (c / np.linalg.norm(c))
    cand = np.isfinite(knn[0])
    truth = d.y[f.test] & cand
    q = truth.any(1)
    big3 = np.argsort(np.where(cand, y_tr.sum(0), -1))[-TOP_K:]
    popular = np.zeros(n_pl, bool)
    popular[big3] = True
    for name, score in (("knn", knn), ("cen", cen)):
        picks = np.zeros_like(truth)
        np.put_along_axis(picks, np.argsort(score, axis=1)[:, -TOP_K:], True, axis=1)
        out[f"recall3_{name}"] = float(((picks & truth).sum(1)[q] / truth.sum(1)[q]).mean())
        if name == "knn":
            small_truth = truth & f.small
            qs = small_truth.any(1)
            out["recall3_small"] = float(((picks & small_truth).sum(1)[qs] / small_truth.sum(1)[qs]).mean())
            out["pop3"] = float((picks & popular).sum(1)[q].mean() / TOP_K)
    out["popular_recall3"] = float(((popular & truth).sum(1)[q] / truth.sum(1)[q]).mean())
    return out


def heldout_scores(f: Fold, z_all: np.ndarray, held: np.ndarray) -> tuple[float, float]:
    """(pair AUC, nn15) for the held-out playlists, which were never in training.
    Pair AUC: pairs whose only shared playlists are held out, against pairs sharing no playlist where at
    least one song is in a held-out playlist. nn15: for test songs in a held-out playlist, the share of
    their 15 nearest training songs that share one of the held-out playlists."""
    y_te = f.d.y[f.test]
    in_h = y_te[:, held].any(1)
    h = y_te[:, held].astype(np.float32)
    rest = np.delete(y_te, held, axis=1).astype(np.float32)
    share_h = (h @ h.T > 0)[f.iu]
    share_rest = (rest @ rest.T > 0)[f.iu]
    z = z_all[f.test]
    ok = ~f.shared_artist
    either = in_h[f.iu[0]] | in_h[f.iu[1]]
    auc = pair_auc((z @ z.T)[f.iu], share_h & ~share_rest & ok, ~f.share & ok & either)
    top = np.argpartition(z @ z_all[f.train].T, -NN_K, axis=1)[:, -NN_K:]
    share_tt = h @ f.d.y[f.train][:, held].T.astype(np.float32) > 0
    return auc, float(np.take_along_axis(share_tt, top, 1)[in_h].mean())


# ---------------------------------------------------------------------------------------------------------
# Experiment


@dataclass
class Config:
    family: str
    label: str
    make: Callable[[Fold, np.ndarray], Transform]
    supervised: bool = False
    results: dict[str, list[float]] = field(default_factory=dict)
    dims: list[int] = field(default_factory=list)
    fit_s: list[float] = field(default_factory=list)


def configs(only: set[str] | None) -> list[Config]:
    def c(family: str, label: str, make, supervised=False, **kw):
        return Config(family, label, lambda f, cols: make(f, cols, **kw), supervised)

    out = [c("base", "cosine on raw vectors", make_base), c("center", "mean-centred", make_center)]
    for dim in (32, 64, 128, 256):
        for alpha in (0.0, 0.5, 1.0):
            out.append(c("pcaw", f"PCA d={dim} alpha={alpha}", make_pcaw, d=dim, alpha=alpha))
    for lam in (0.1, 1.0, 10.0):
        for balance in (False, True):
            for rank in (None, 16):
                label = f"ridge lam={lam} {'std' if balance else 'raw'} rank={rank or 'all'}"
                out.append(c("ridge", label, make_ridge, True, lam=lam, balance=balance, rank=rank))
    for dim in (32, 64, 128, 200):
        for beta in (0.3, 0.5, 0.7):
            label = f"blend PCA d={dim} a=0.5 beta={beta} + ridge lam=1 std"
            out.append(
                c("blend", label, make_blend, True, d=dim, alpha=0.5, beta=beta, lam=1.0, balance=True, rank=None)
            )
    for dim in (32, 64, 128, 256):
        for balance in (False, True):
            label = f"contrastive d={dim} {'balanced' if balance else 'plain'}"
            out.append(c("contra", label, make_contra, True, d=dim, steps=60, lr=3e-3, balance=balance))
    for hidden in (256, 512):
        out.append(
            c(
                "mlp",
                f"MLP head d=64 hidden={hidden} balanced",
                make_contra,
                True,
                d=64,
                steps=60,
                lr=3e-3,
                hidden=hidden,
            )
        )
    return [cfg for cfg in out if only is None or cfg.family in only or cfg.family == "base"]


def run_sweep(d: Data, cfgs: list[Config], seeds: Sequence[int], n_folds: int) -> int:
    n_runs = 0
    for seed in seeds:
        for test in folds_by_artist(d.labelled, d.primary, n_folds, seed):
            f = Fold(d, test)
            t0 = time.time()
            for cfg in cfgs:
                tic = time.time()
                z = cfg.make(f, f.cols)(d.x)
                cfg.fit_s.append(time.time() - tic)
                cfg.dims.append(z.shape[1])
                for k, v in evaluate(f, z).items():
                    cfg.results.setdefault(k, []).append(v)
            n_runs += 1
            print(
                f"seed {seed} fold {n_runs}: {len(test)} test, {len(f.train)} train songs, {time.time() - t0:.0f}s",
                flush=True,
            )
    return n_runs


def run_heldout(d: Data, cfgs: list[Config], seeds: Sequence[int], n_folds: int, n_groups: int = 5) -> dict[str, dict]:
    """Hold out whole playlists: split the playlists with >= 8 songs into groups, and for each group train
    without its labels (as if those playlists did not exist yet), then score pairs inside it."""
    big = np.flatnonzero(d.size >= 8)
    stats = {cfg.label: {"unseen": [], "seen": [], "unseen_nn": [], "seen_nn": []} for cfg in cfgs}
    for seed in seeds:
        order = np.random.default_rng(100 + seed).permutation(big)
        groups = [order[g::n_groups] for g in range(n_groups)]
        for test in folds_by_artist(d.labelled, d.primary, n_folds, seed):
            f = Fold(d, test)
            for held in groups:
                keep = np.setdiff1d(f.cols, held)
                for cfg in cfgs:
                    for kind, cols in (("unseen", keep), ("seen", f.cols)):
                        if kind == "seen" and not cfg.supervised:
                            continue
                        z = cfg.make(f, cols)(d.x)
                        auc, nn = heldout_scores(f, z, held)
                        stats[cfg.label][kind].append(auc)
                        stats[cfg.label][kind + "_nn"].append(nn)
            print(f"heldout seed {seed}: fold done", flush=True)
    return stats


def summarise(cfgs: list[Config], n_per_seed: int) -> list[dict]:
    base = cfgs[0].results
    rows = []
    for cfg in cfgs:
        row: dict = {
            "family": cfg.family,
            "config": cfg.label,
            "dim": float(np.mean(cfg.dims)),
            "fit_s": float(np.mean(cfg.fit_s)),
        }
        for k, v in cfg.results.items():
            v = np.array(v)
            row[k] = float(np.nanmean(v))
            row[k + "_sd"] = float(np.nanstd(v))
            diff = v - np.array(base[k])
            row[k + "_d"] = float(np.nanmean(diff))
            row[k + "_dsd"] = float(np.nanstd(diff))
            # Folds of one seed are the independent-ish units; repeats of the same songs are not.
            row[k + "_real"] = bool(abs(row[k + "_d"]) > 2 * row[k + "_dsd"] / np.sqrt(n_per_seed))
        rows.append(row)
    return rows


COLS = [
    ("pair_auc", "pair AUC"),
    ("recall3_knn", "recall@3"),
    ("nn15", "nn15"),
    ("nn15_lift", "nn15 over song's base rate"),
    ("pair_auc_small", "AUC small"),
    ("recall3_small", "recall@3 small"),
    ("nn15_small", "nn15 small"),
    ("pop3", "pop3"),
]


def table(rows: list[dict]) -> str:
    head = "| config | dim | " + " | ".join(n for _, n in COLS) + " |"
    lines = [head, "|---|---:|" + "---:|" * len(COLS)]
    for r in rows:
        cells = []
        for k, _ in COLS:
            mark = "" if r["family"] == "base" else (" *" if r[k + "_real"] else "")
            delta = "" if r["family"] == "base" else f" ({r[k + '_d']:+.3f}){mark}"
            cells.append(f"{r[k]:.3f}±{r[k + '_sd']:.3f}{delta}")
        lines.append(f"| {r['config']} | {r['dim']:.0f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_against_metrics(d: Data) -> None:
    """The vectorised pair AUC must equal bench/metrics.pair_auc on a subset."""
    from metrics import pair_auc as reference

    f = Fold(d, folds_by_artist(d.labelled, d.primary, 5, 0)[0])
    sub = f.test[:150]
    keys = [d.keys[int(i)] for i in sub]
    labels = {k: {str(j) for j in np.flatnonzero(d.y[int(i)])} for k, i in zip(keys, sub)}
    artists = {k: set(d.artists[int(i)]) for k, i in zip(keys, sub)}
    ref = reference(keys, d.x[sub], labels, artists, True)
    iu = np.triu_indices(len(sub), 1)
    y = d.y[sub].astype(np.float32)
    shared = (y @ y.T > 0)[iu]
    inc = np.array(
        [[a in artists[k] for a in sorted({x for v in artists.values() for x in v})] for k in keys], np.float32
    )
    art = (inc @ inc.T > 0)[iu]
    mine = pair_auc((d.x[sub] @ d.x[sub].T)[iu], shared & ~art, ~shared & ~art)
    print(f"check: metrics.pair_auc {ref:.6f} vs vectorised {mine:.6f}")
    assert abs(ref - mine) < 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3, help="repeats of the artist split")
    ap.add_argument("--only", help="comma-separated families to run besides base: center,pcaw,ridge,blend,contra,mlp")
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    d = load(args.data_dir)
    print(
        f"{len(d.keys)} songs, {len(d.labelled)} in a counted playlist, {d.y.shape[1]} playlists "
        f"(sizes {sorted(d.size.tolist())}), {int((d.size < SMALL).sum())} under {SMALL}"
    )
    if args.check:
        check_against_metrics(d)
    only = set(args.only.split(",")) if args.only else None
    cfgs = configs(only)
    seeds = list(range(args.seeds))
    n_runs = run_sweep(d, cfgs, seeds, args.folds)
    rows = summarise(cfgs, args.folds)
    print("\nMean ± sd over folds; (delta vs base), * = delta beyond 2 sd/sqrt(folds per split)\n")
    print(table(rows))
    print(
        "\nfold sizes: popular-playlists baseline recall@3 "
        f"{np.mean(cfgs[0].results['popular_recall3']):.3f} over {n_runs} folds; chance share of neighbours "
        "is the base-rate of sharing a playlist."
    )

    held = {}
    if not args.no_heldout:
        best = {}
        for r, cfg in zip(rows, cfgs):
            if cfg.family not in ("base", "center") and (cfg.family not in best or r["pair_auc"] > best[cfg.family][0]):
                best[cfg.family] = (r["pair_auc"], cfg)
        picks = [cfgs[0]] + [cfg for _, cfg in best.values()]
        held = run_heldout(d, picks, seeds[: max(1, len(seeds) - 1)], args.folds)
        print("\nHeld-out playlists (never in training): mean±sd over folds x playlist groups; (delta vs base)\n")
        print("| config | pair AUC unseen | pair AUC if seen | nn15 unseen | nn15 if seen |\n|---|---:|---:|---:|---:|")
        base_h = held[cfgs[0].label]

        def cell(v: list[float], ref: list[float] | None) -> str:
            if not v:
                return "n/a"
            delta = (
                ""
                if ref is None
                else f" ({np.nanmean(np.array(v) - np.array(ref)):+.3f}±{np.nanstd(np.array(v) - np.array(ref)):.3f})"
            )
            return f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}{delta}"

        for label, s in held.items():
            ref = label != cfgs[0].label
            print(
                f"| {label} | {cell(s['unseen'], base_h['unseen'] if ref else None)} | {cell(s['seen'], base_h['unseen'] if ref else None)} "
                f"| {cell(s['unseen_nn'], base_h['unseen_nn'] if ref else None)} | {cell(s['seen_nn'], base_h['unseen_nn'] if ref else None)} |"
            )

    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "heldout": held}, indent=1, default=float))


if __name__ == "__main__":
    main()
