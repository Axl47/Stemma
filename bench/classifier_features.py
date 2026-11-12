"""Does the playlist classifier place new-artist songs better with more features than the whole-mix sound vector?

Usage (from the Stemma repo, one BLAS thread budget of 4 so it can share the machine):
    uv run --no-sync python bench/classifier_features.py <cento_data_dir> [--seeds 0 1] [--weights 0.5 1.0] [--sets ...]

Inputs are exactly what `audio-embed classify` gets from Cento (`classifier-input.jsonl` labels, `vectors.jsonl`),
plus optional extra blocks that are appended to the design matrix with a weight:
  vocals / rest   stem vectors from `~/.cache/audio-embed-engine/stem-vectors/<key>.npz` (key: upper ISRC, else Spotify id)
  themes          Cento `themes.json`, 40 lyric-theme scores (std devs within the song's language)
  lyrics          Cento `lyrics-index.json` + `.bin`: mean of a song's passage vectors (multilingual-e5-base)
  patch_std/max   per-dimension std / max over the song's ~2 s patch vectors (`patches/patches.f32`)

Protocol = classifier_tuning.py: labelled songs only, 5 folds split by primary artist (training songs sharing any artist
with a test song are dropped), a ridge model per playlist, Platt curves and AUC trust fitted on held-out scores *inside*
the training songs, then everything is judged on the test fold with the shipped rule (confidence 0.9, min AUC 0.9, at most
one pick per song). Nothing is fitted on test songs: the language columns, and the mean that stands in for a song missing
a block, come from the training songs of that fold. A song missing a block is scored from the other blocks (its block is
the training mean, so it adds nothing after centering).

Results per (feature set, seed) are cached, so reruns only compute what is new. Delete the cache if the data changes.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"  # the machine is shared with another experiment

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audio_embed_engine.classifier import (
    CONFIDENCE,
    MIN_TRUST,
    auc,
    clean_train,
    folds_by_artist,
    out_of_fold,
    platt,
    score_ridge,
    with_block,
    with_language,
)

CACHE = Path.home() / ".cache" / "audio-embed-engine"
STEMS = CACHE / "stem-vectors"
PATCHES = CACHE / "patches"
WORK = CACHE / "classifier-features"
MIN_SIZE = 12
KS = (1, 3, 5)
# Feature sets to screen: name -> blocks. Each block gets the weight from --weights.
SETS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "+vocals": ("vocals",),
    "+rest": ("rest",),
    "+vocals+rest": ("vocals", "rest"),
    "+themes": ("themes",),
    "+lyrics": ("lyrics",),
    "+patch_std": ("patch_std",),
    "+patch_max": ("patch_max",),
}


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def patch_stats() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per song: (mean, std, max) over its patch rows. One pass over the 2.7 GB file, cached."""
    out = WORK / "patch-stats.npz"
    if not out.exists():
        rows = [json.loads(line) for line in open(PATCHES / "index.jsonl")]
        total = max(r["start"] + r["n"] for r in rows)
        patches = np.memmap(PATCHES / "patches.f32", np.float32, "r", shape=(total, 1280))
        keys, mean, std, mx = [], [], [], []
        for r in rows:
            p = np.asarray(patches[r["start"] : r["start"] + r["n"]])
            keys.append(r["key"])
            mean.append(p.mean(0))
            std.append(p.std(0))
            mx.append(p.max(0))
        np.savez(out, keys=np.array(keys), mean=np.array(mean), std=np.array(std), max=np.array(mx))
    z = np.load(out)
    return {k: (m, s, x) for k, m, s, x in zip(z["keys"], z["mean"], z["std"], z["max"], strict=True)}


class Data:
    """The classifier's songs, labels and every optional feature block, row-aligned."""

    def __init__(self, cento: Path):
        vectors = {}
        for line in open(cento / "data" / "vectors.jsonl"):
            row = json.loads(line)
            if row.get("vector"):
                vectors[row["key"]] = row["vector"]
        songs: dict[str, dict] = {}
        for line in open(cento / "data" / "classifier-input.jsonl"):
            row = json.loads(line)
            if row["key"] in vectors:
                songs.setdefault(row["key"], row)
        self.keys = list(songs)
        counts: dict[str, int] = {}
        for s in songs.values():
            for label in s["labels"]:
                counts[label] = counts.get(label, 0) + 1
        self.playlists = sorted(p for p, n in counts.items() if n >= MIN_SIZE)
        column = {p: i for i, p in enumerate(self.playlists)}
        self.y = np.zeros((len(self.keys), len(self.playlists)), bool)
        for i, k in enumerate(self.keys):
            for label in songs[k]["labels"]:
                if label in column:
                    self.y[i, column[label]] = True
        self.mix = np.array([vectors[k] for k in self.keys], np.float32)
        self.artists = [frozenset(songs[k]["artists"]) or frozenset({k}) for k in self.keys]
        self.primary = [(songs[k]["artists"] or [k])[0] for k in self.keys]
        self.languages = [songs[k].get("language") for k in self.keys]
        self.labelled = np.flatnonzero(self.y.any(1))

        isrc = {}
        for line in open(cento / "data" / "tracks.jsonl"):
            t = json.loads(line)
            isrc[t["key"]] = t["isrc"]
        # the engine's key for a song: upper ISRC, else the Spotify id
        self.ckeys = [(isrc.get(k) or k).upper() for k in self.keys]
        self.blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._stems()
        self._lyrics(cento)
        self._patches()

    def _add(self, name: str, rows: dict[str, np.ndarray], dim: int) -> None:
        arr = np.zeros((len(self.keys), dim), np.float32)
        mask = np.zeros(len(self.keys), bool)
        for i, k in enumerate(self.ckeys):
            if k in rows:
                arr[i], mask[i] = rows[k], True
        self.blocks[name] = (arr, mask)

    def _stems(self) -> None:
        vocals, rest = {}, {}
        for k in set(self.ckeys):
            f = STEMS / f"{k}.npz"
            if f.exists():
                z = np.load(f)
                vocals[k], rest[k] = z["vocals"], z["rest"]
        for name, rows in (("vocals", vocals), ("rest", rest)):
            self._add(name, {k: v / max(float(np.linalg.norm(v)), 1e-9) for k, v in rows.items()}, 1280)

    def _lyrics(self, cento: Path) -> None:
        themes = json.load(open(cento / "data" / "themes.json"))
        n = len(themes["themes"])
        # divide by sqrt(n): unit-variance scores then make a row of about unit length, like the sound vectors
        self._add(
            "themes", {s["key"].upper(): np.array(s["scores"], np.float32) / np.sqrt(n) for s in themes["songs"]}, n
        )
        index = json.load(open(cento / "data" / "lyrics-index.json"))
        vec = np.fromfile(cento / "data" / "lyrics-index.bin", np.float32).reshape(-1, index["dim"])
        by_song: dict[str, list[int]] = {}
        for row, k in enumerate(index["keys"]):
            by_song.setdefault(k.upper(), []).append(row)
        self._add("lyrics", {k: unit(vec[r].mean(0, keepdims=True))[0] for k, r in by_song.items()}, index["dim"])

    def _patches(self) -> None:
        stats = patch_stats()
        self._add("patch_std", {k: v[1] / max(float(np.linalg.norm(v[1])), 1e-9) for k, v in stats.items()}, 1280)
        self._add("patch_max", {k: v[2] / max(float(np.linalg.norm(v[2])), 1e-9) for k, v in stats.items()}, 1280)

    def coverage(self) -> str:
        lab = np.zeros(len(self.keys), bool)
        lab[self.labelled] = True
        lines = [f"{len(self.keys)} songs, {len(self.labelled)} labelled, {len(self.playlists)} playlists"]
        for name, (_, mask) in self.blocks.items():
            lines.append(
                f"  {name:<10} {int(mask.sum()):>5} songs ({mask.mean():.1%}), {int((mask & lab).sum()):>5} labelled ({(mask & lab).sum() / lab.sum():.1%})"
            )
        return "\n".join(lines)

    def design(self, train: np.ndarray, spec: list[tuple[str, float]]) -> np.ndarray:
        """Feature matrix for every song, with everything data-dependent fitted on `train` rows only."""
        in_train = np.zeros(len(self.keys), bool)
        in_train[train] = True
        x, _ = with_language(self.mix, self.languages, in_train)
        for name, weight in spec:
            arr, mask = self.blocks[name]
            x = with_block(x, arr, mask, in_train, weight=weight)  # the shipped step: absent rows get the training mean
        return x


def run_seed(d: Data, spec: list[tuple[str, float]], seed: int) -> dict[str, np.ndarray]:
    """One pass over 5 artist folds; per labelled song: R@k, whether it was auto-picked, and whether the pick was right."""
    idx, y = d.labelled, d.y
    pos = {int(i): j for j, i in enumerate(idx)}
    n = len(idx)
    out = {f"r{k}": np.zeros(n) for k in KS} | {
        "picked": np.zeros(n),
        "right": np.zeros(n),
        "recall": np.zeros(n),
        "fold": np.zeros(n, int),
    }
    for f, test in enumerate(folds_by_artist(idx, d.primary, 5, seed + 100)):
        train = clean_train(np.setdiff1d(idx, test), test, d.artists)
        x = d.design(train, spec)
        scores = score_ridge(x[test], x[train], y[train].astype(float))
        oof = out_of_fold(x, y, train, d.artists, d.primary, folds=5, repeats=2, seed=seed + 200)
        prob = np.zeros(scores.shape)
        for p in range(y.shape[1]):
            s, t = oof[:, p], y[train, p]
            trust = auc(s, t)
            a, b = platt(s, t) if t.sum() >= 5 else (0.0, 0.0)
            if not np.isnan(trust) and trust >= MIN_TRUST and a > 0:
                prob[:, p] = 1 / (1 + np.exp(-np.clip(a * scores[:, p] + b, -30, 30)))
        rows = np.array([pos[int(i)] for i in test])
        truth = y[test]
        order = np.argsort(-scores, axis=1)
        for k in KS:
            top = np.take_along_axis(truth, order[:, :k], axis=1).sum(1)
            out[f"r{k}"][rows] = top / truth.sum(1)
        best = prob.argmax(1)
        picked = prob[np.arange(len(test)), best] >= CONFIDENCE
        right = picked & truth[np.arange(len(test)), best]
        out["picked"][rows] = picked
        out["right"][rows] = right
        out["recall"][rows] = right / truth.sum(1)
        out["fold"][rows] = f
    return out


def cached_run(d: Data, label: str, spec: list[tuple[str, float]], seed: int) -> dict[str, np.ndarray]:
    path = WORK / "runs" / f"{label}__s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    t0 = time.time()
    out = run_seed(d, spec, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"  {label} seed {seed}: R@3 {out['r3'].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---- summaries ---------------------------------------------------------------------------------------------------

METRICS = ("R@1", "R@3", "R@5", "pick precision", "pick recall", "picks/song")


def per_song(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per labelled song, summed (picks) or averaged (rest) over seeds."""
    m = lambda key: np.mean([r[key] for r in runs], axis=0)
    return {
        "r1": m("r1"),
        "r3": m("r3"),
        "r5": m("r5"),
        "picked": m("picked"),
        "right": m("right"),
        "recall": m("recall"),
    }


def values(s: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
    """The six metrics, with song weights `w` (artist-bootstrap multiplicities)."""
    tot = w.sum()
    picked = (w * s["picked"]).sum()
    return np.array([
        (w * s["r1"]).sum() / tot, (w * s["r3"]).sum() / tot, (w * s["r5"]).sum() / tot,
        (w * s["right"]).sum() / picked if picked else np.nan,
        (w * s["recall"]).sum() / tot, picked / tot,
    ])  # fmt: skip


def bootstrap(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    artist: np.ndarray,
    only: np.ndarray | None = None,
    draws: int = 1000,
    seed: int = 0,
):
    """95% interval of (b - a) for every metric, resampling whole artists with the same draw for both.
    `only` restricts the songs counted (a 0/1 mask)."""
    rng = np.random.default_rng(seed)
    n_art = artist.max() + 1
    deltas = np.empty((draws, len(METRICS)))
    for i in range(draws):
        mult = np.bincount(rng.integers(0, n_art, n_art), minlength=n_art)[artist].astype(float)
        if only is not None:
            mult = mult * only
        deltas[i] = values(b, mult) - values(a, mult)
    return np.nanpercentile(deltas, [2.5, 97.5], axis=0)


def fold_noise(runs: list[dict[str, np.ndarray]], base: list[dict[str, np.ndarray]]) -> tuple[float, float]:
    """SD over (seed, fold) cells of the R@3 mean, and SD of the paired (variant - baseline) cell difference."""
    cells, diffs = [], []
    for r, b in zip(runs, base, strict=True):
        for f in range(5):
            m = r["fold"] == f
            cells.append(r["r3"][m].mean())
            diffs.append(r["r3"][m].mean() - b["r3"][m].mean())
    return float(np.std(cells, ddof=1)), float(np.std(diffs, ddof=1))


def covered(d: Data, label: str) -> np.ndarray | None:
    """Labelled songs that have every block in the set (None for the baseline)."""
    names = [b for b, _ in spec_of(parse_blocks(label.split(" @")[0]), 1.0)]
    if not names:
        return None
    return np.all([d.blocks[n][1][d.labelled] for n in names], axis=0).astype(float)


def table(d: Data, results: dict[str, list[dict[str, np.ndarray]]]) -> str:
    base = results["baseline"]
    labels = {p: i for i, p in enumerate(sorted(set(d.primary)))}
    artist = np.array([labels[d.primary[int(i)]] for i in d.labelled])
    sb = per_song(base)
    ones = np.ones(len(d.labelled))
    lines = [
        f"| feature set | {' | '.join(METRICS)} | ΔR@3 vs baseline (95% CI) | ΔR@3 on songs that have it | fold SD of R@3 | SD of paired fold Δ |",
        f"|---|{'---|' * (len(METRICS) + 4)}",
    ]
    for name, runs in results.items():
        s = per_song(runs)
        v = values(s, ones)
        fmt = [f"{x:.1%}" if i != 5 else f"{x:.3f}" for i, x in enumerate(v)]
        sd, sd_pair = fold_noise(runs, base)
        cov = covered(d, name)
        on = "-"
        if name == "baseline":
            delta = "-"
        else:
            if cov is not None and cov.mean() < 1:
                lo2, hi2 = bootstrap(sb, s, artist, cov)[:, 1]
                dv2 = values(s, cov)[1] - values(sb, cov)[1]
                on = f"{dv2 * 100:+.1f} pp ({lo2 * 100:+.1f} to {hi2 * 100:+.1f}), n={int(cov.sum())}"
            lo, hi = bootstrap(sb, s, artist)[:, 1]
            dv = v[1] - values(sb, ones)[1]
            delta = f"{dv * 100:+.1f} pp ({lo * 100:+.1f} to {hi * 100:+.1f})"
            delta += " **real**" if lo > 0 or hi < 0 else " noise"
        lines.append(f"| {name} | {' | '.join(fmt)} | {delta} | {on} | {sd * 100:.1f} pp | {sd_pair * 100:.1f} pp |")
    return "\n".join(lines)


def parse_blocks(name: str) -> tuple[str, ...]:
    """A SETS name, or ad hoc 'lyrics:3+vocals:0.5' (block[:weight], joined by +)."""
    return SETS[name] if name in SETS else tuple(t for t in name.lstrip("+").split("+") if t)


def spec_of(blocks: tuple[str, ...], default: float) -> list[tuple[str, float]]:
    return [(b.split(":")[0], float(b.split(":")[1]) if ":" in b else default) for b in blocks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cento", type=Path, help="Cento repo root (read only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0])
    ap.add_argument("--sets", nargs="*", help="names from SETS, or ad hoc 'vocals+themes'; default all")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    d = Data(a.cento)
    print(d.coverage(), flush=True)

    wanted = a.sets or list(SETS)
    results: dict[str, list[dict[str, np.ndarray]]] = {}
    for name in ["baseline", *[w for w in wanted if w != "baseline"]]:
        blocks = parse_blocks(name)
        explicit = any(":" in b for b in blocks)
        for w in [1.0] if not blocks or explicit else a.weights:
            label = name if not blocks or explicit else f"{name} @{w:g}"
            spec = spec_of(blocks, w)
            file_label = label.replace(" ", "_").replace("+", "_").replace("@", "w").replace(":", "w")
            results[label] = [cached_run(d, file_label, spec, s) for s in a.seeds]
    print(
        f"\nseeds {a.seeds}, {len(d.labelled)} labelled songs, artist-held-out, shipped rule (confidence {CONFIDENCE}, min AUC {MIN_TRUST})\n"
    )
    print(table(d, results))


if __name__ == "__main__":
    main()
