"""Does adding per-song signal features (tempo, loudness, vocals) to the playlist classifier help?

Usage (from the Stemma repo; the cento arg is read only):
    PYTHONPATH=bench uv run --group bench python bench/classifier_signal.py <cento repo> [--seeds 10 11 12] [--out report.md]

Reuses bench/classifier_features.py (data loading, artist-held-out folds, ridge scoring, `with_block`) and compares
  baseline = sound + language + lyrics@2 (the shipped classifier)
  baseline + a small numeric "signal" block from ~/.cache/audio-embed-engine/features/features.jsonl
at several weights, plus tempo-only / loudness-only / vocals-only column groups.

The signal block is standardised on the training fold only (mean/std over training songs that have features), a
missing value becomes 0 after standardising, and a "has features" column is appended. The block is divided by
sqrt(#columns) before the weight, so a row has about unit length like the other blocks.

Differences from classifier_features.py, which only reports the metrics below for held-out artists:
  - Only R@k is computed here (ranking by ridge score). The Platt/AUC "auto-check" machinery is skipped; it does not
    change R@k. R@k reproduces the cached classifier_features numbers (checked with --verify).
  - "Known artist" numbers are new: 5 random song folds (not split by artist), trained on all other labelled songs,
    so most test songs have artist-mates in training.
Results are cached per (label, seed) in ~/.cache/audio-embed-engine/classifier-signal/.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from classifier_features import KS, Data
from audio_embed_engine.classifier import (
    clean_train,
    folds_by_artist,
    score_ridge,
    with_block,
    with_language,
)

FEATURES = Path.home() / ".cache" / "audio-embed-engine" / "features" / "features.jsonl"
WORK = Path.home() / ".cache" / "audio-embed-engine" / "classifier-signal"
LYRICS_WEIGHT = 2.0
WEIGHTS = (0.25, 0.5, 1.0, 2.0)

# column -> group; groups are the ablations ("all" is every column)
COLUMNS: dict[str, str] = {
    "bpm_fold": "tempo",
    "log2_bpm": "tempo",
    "reg": "tempo",
    "bpb_not4": "tempo",
    "lufs": "loudness",
    "lra": "loudness",
    "level_t": "loudness",
    "tail_len": "loudness",
    "tail_drop": "loudness",
    "voc_share": "vocals",
    "voc_energy": "vocals",
    "voc_first": "vocals",
    "mode": "tonal",
    "key_strength": "tonal",
    "log_dur": "duration",
}
GROUPS = ("all", "tempo", "loudness", "vocals")


def song_columns(row: dict) -> dict[str, float]:
    """The numeric columns of one features.jsonl row (NaN where the extractor gave nothing)."""
    nan = float("nan")
    t, ld, v, tn = row.get("tempo") or {}, row.get("loud") or {}, row.get("voc") or {}, row.get("tonal") or {}
    tail = ld.get("tail")
    all_key = tn.get("all")
    bpm = t.get("bpm")

    def num(x):
        return nan if x is None else float(x)

    return {
        "bpm_fold": num(t.get("bpm_fold")),
        "log2_bpm": float(np.log2(bpm)) if bpm else nan,
        "reg": num(t.get("reg")),
        "bpb_not4": float(t["bpb"] != 4) if t.get("bpb") else nan,
        "lufs": num(ld.get("lufs")),
        "lra": num(ld.get("lra")),
        "level_t": num(ld.get("level_t")),
        "tail_len": float(np.log1p(tail["len"])) if tail else (0.0 if ld else nan),
        "tail_drop": float(tail["drop"]) if tail else (0.0 if ld else nan),
        "voc_share": num(v.get("share")),
        "voc_energy": num(v.get("energy")),
        "voc_first": num(v.get("first")),
        "mode": (1.0 if all_key[1] == "major" else 0.0) if all_key else nan,
        "key_strength": float(all_key[2]) if all_key else nan,
        "log_dur": float(np.log(row["dur"])) if row.get("dur") else nan,
    }


class Signal(Data):
    """Data plus the raw (unstandardised) signal matrix, row-aligned with the songs."""

    def __init__(self, cento: Path):
        super().__init__(cento)
        by_key = {}
        for line in open(FEATURES):
            row = json.loads(line)
            if "error" not in row:
                by_key[row["key"].upper()] = song_columns(row)
        self.names = list(COLUMNS)
        self.raw = np.full((len(self.keys), len(self.names)), np.nan)
        self.has = np.zeros(len(self.keys), bool)
        for i, k in enumerate(self.ckeys):
            if k in by_key:
                self.raw[i] = [by_key[k][c] for c in self.names]
                self.has[i] = True

    def signal_block(self, in_train: np.ndarray, group: str) -> np.ndarray:
        """Standardised on training songs, missing -> 0, plus the has-features column, / sqrt(#columns)."""
        cols = [j for j, c in enumerate(self.names) if group == "all" or COLUMNS[c] == group]
        raw = self.raw[:, cols]
        fit = raw[in_train & self.has]
        mu, sd = np.nanmean(fit, 0), np.nanstd(fit, 0)
        z = (raw - mu) / np.where(sd > 1e-9, sd, 1.0)
        z = np.nan_to_num(z, nan=0.0)
        block = np.hstack([z, self.has[:, None].astype(float)])
        return (block / np.sqrt(block.shape[1])).astype(np.float32)

    def design(self, train: np.ndarray, spec: tuple[str, float] | None) -> np.ndarray:
        in_train = np.zeros(len(self.keys), bool)
        in_train[train] = True
        x, _ = with_language(self.mix, self.languages, in_train)
        arr, mask = self.blocks["lyrics"]
        x = with_block(x, arr, mask, in_train, weight=LYRICS_WEIGHT)
        if spec is not None:
            group, weight = spec
            x = np.hstack([x, weight * self.signal_block(in_train, group)]).astype(x.dtype)
        return x


def hits(scores: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per test song: R@k, and per (song, playlist) whether a true playlist is in the top 3."""
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in KS:
        top = np.take_along_axis(truth, order[:, :k], axis=1)
        out[f"r{k}"] = top.sum(1) / truth.sum(1)
    top3 = np.zeros_like(truth)
    np.put_along_axis(top3, order[:, :3], True, axis=1)
    out["hit3"] = top3 & truth
    return out


def run(d: Signal, spec: tuple[str, float] | None, seed: int, mode: str) -> dict[str, np.ndarray]:
    """One pass of 5 folds. mode 'held' = split by artist, training drops artist-mates (the classifier_features
    protocol); 'known' = random song folds, nothing dropped. Arrays are indexed by position in d.labelled."""
    idx, y = d.labelled, d.y
    pos = {int(i): j for j, i in enumerate(idx)}
    n = len(idx)
    out = {f"r{k}": np.zeros(n) for k in KS} | {"hit3": np.zeros((n, y.shape[1]), bool), "tested": np.zeros(n, bool)}
    if mode == "held":
        folds = folds_by_artist(idx, d.primary, 5, seed + 100)
    else:
        order = np.random.default_rng(seed + 300).permutation(idx)
        folds = [np.sort(order[f::5]) for f in range(5)]
    for test in folds:
        train = clean_train(np.setdiff1d(idx, test), test, d.artists) if mode == "held" else np.setdiff1d(idx, test)
        x = d.design(train, spec)
        h = hits(score_ridge(x[test], x[train], y[train].astype(float)), y[test])
        rows = np.array([pos[int(i)] for i in test])
        if mode == "known":  # only songs that share an artist with some training song
            seen = set().union(*(d.artists[int(i)] for i in train))
            keep = np.array([bool(d.artists[int(i)] & seen) for i in test])
            rows, h = rows[keep], {k: v[keep] for k, v in h.items()}
        for k, v in h.items():
            out[k][rows] = v
        out["tested"][rows] = True
    return out


def cached(d: Signal, label: str, spec: tuple[str, float] | None, seed: int, mode: str) -> dict[str, np.ndarray]:
    path = WORK / f"{label}__{mode}__s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    t0 = time.time()
    out = run(d, spec, seed, mode)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"  {label} {mode} seed {seed}: R@3 {out['r3'][out['tested']].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def fmt(x: float) -> str:
    return f"{x * 100:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cento", type=Path, help="Cento repo root (read only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    ap.add_argument("--out", type=Path, help="also write the tables here")
    ap.add_argument("--verify", action="store_true", help="check R@3 against classifier_features' cached lyrics@2 runs")
    a = ap.parse_args()
    d = Signal(a.cento)
    lab = d.labelled
    print(f"{len(d.keys)} songs, {len(lab)} labelled, {len(d.playlists)} playlists; "
          f"signal features for {int(d.has.sum())} songs ({d.has[lab].mean():.1%} of labelled)", flush=True)
    print("missing per column (labelled songs with features):",
          {c: int(np.isnan(d.raw[lab][d.has[lab], j]).sum()) for j, c in enumerate(d.names)}, flush=True)

    configs: dict[str, tuple[str, float] | None] = {"baseline": None}
    for g in GROUPS:
        for w in WEIGHTS:
            configs[f"signal:{g} @{w:g}"] = (g, w)
    res = {(name, mode): [cached(d, name.replace(" ", "_").replace(":", "-"), spec, s, mode) for s in a.seeds]
           for name, spec in configs.items() for mode in ("held", "known")}

    if a.verify:  # classifier_features cached lyrics@2 (seeds 0-4) vs this module's baseline
        old = Path.home() / ".cache/audio-embed-engine/classifier-features/runs"
        for s in (0, 1):
            f = old / f"_lyrics_w2__s{s}.npz"
            if f.exists():
                mine = cached(d, "baseline", None, s, "held")
                print(f"verify seed {s}: cached R@3 {np.load(f)['r3'].mean():.4f} vs here {mine['r3'].mean():.4f}")

    lines = []
    n_held = int(res[("baseline", "held")][0]["tested"].sum())
    n_known = int(res[("baseline", "known")][0]["tested"].sum())
    artists = {p: i for i, p in enumerate(sorted({d.primary[int(i)] for i in lab}))}
    art = np.array([artists[d.primary[int(i)]] for i in lab])

    def per_seed(name: str, mode: str, key: str) -> np.ndarray:
        return np.array([r[key][r["tested"]].mean() for r in res[(name, mode)]])

    def boot_ci(name: str, mode: str, draws: int = 1000) -> tuple[float, float]:
        """95% artist bootstrap of the R@3 gain over baseline (seed-averaged per song, same draw for both)."""
        b = np.mean([r["r3"] for r in res[("baseline", mode)]], 0)
        v = np.mean([r["r3"] for r in res[(name, mode)]], 0)
        t = res[("baseline", mode)][0]["tested"]
        rng = np.random.default_rng(0)
        n_art = art.max() + 1
        ds = []
        for _ in range(draws):
            m = np.bincount(rng.integers(0, n_art, n_art), minlength=n_art)[art] * t
            ds.append(((v - b) * m).sum() / m.sum())
        return tuple(np.percentile(ds, [2.5, 97.5]))

    for mode, title, n in (("held", "Held-out artists (artist-split folds, training drops artist-mates)", n_held),
                           ("known", "Known artists (random song folds, artist-mates stay in training)", n_known)):
        lines += [f"### {title}", "", f"{n} test songs per seed, {len(a.seeds)} seeds ({a.seeds[0]}..{a.seeds[-1]}). "
                  "Mean over seeds; ± is the SD across seeds. Δ is paired with the baseline on the same folds: "
                  "mean Δ over seeds, SD of Δ across seeds, and a 95% bootstrap interval over artists.", "",
                  "| config | R@1 | R@3 | R@5 | ΔR@1 (pp) | ΔR@3 (pp) | SD of ΔR@3 | 95% CI ΔR@3 | ΔR@5 (pp) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in configs:
            cells = [f"{fmt(per_seed(name, mode, f'r{k}').mean())} ± {fmt(per_seed(name, mode, f'r{k}').std(ddof=1))}"
                     for k in KS]
            if name == "baseline":
                lines.append(f"| baseline | {' | '.join(cells)} | - | - | - | - | - |")
                continue
            dl = {k: per_seed(name, mode, f"r{k}") - per_seed("baseline", mode, f"r{k}") for k in KS}
            lo, hi = boot_ci(name, mode)
            lines.append(f"| {name} | {' | '.join(cells)} | {fmt(dl[1].mean())} | {fmt(dl[3].mean())} | "
                         f"{fmt(dl[3].std(ddof=1))} | {fmt(lo)} to {fmt(hi)} | {fmt(dl[5].mean())} |")
        lines.append("")

    # per-playlist: R@3 per playlist (true members in top 3), held-out artists, seed-averaged
    best = max((c for c in configs if c != "baseline"), key=lambda c: per_seed(c, "held", "r3").mean())
    lines += [f"### Per playlist, held-out artists: baseline → `{best}` (the highest held-out R@3 config)", "",
              "Playlist recall@3 = share of the playlist's songs (as test songs) that have the playlist in their top 3. "
              "Seed-averaged. SD = spread of the change across seeds. These are small groups: a change under ~5 pp is "
              "within noise.", "",
              "| playlist | n | before → after (%) | Δ pp | SD of Δ |", "|---|---|---|---|---|"]
    n_pl = d.y[lab].sum(0)
    names = {p["id"]: p["name"] for p in json.load(open(a.cento / "data" / "library.json"))["playlists"]}
    base_h = np.array([r["hit3"].sum(0) / n_pl for r in res[("baseline", "held")]])
    best_h = np.array([r["hit3"].sum(0) / n_pl for r in res[(best, "held")]])
    delta = best_h - base_h
    for j in np.argsort(-np.abs(delta.mean(0)))[:10]:
        lines.append(f"| {names.get(d.playlists[j], d.playlists[j])} | {int(n_pl[j])} | {fmt(base_h[:, j].mean())} → {fmt(best_h[:, j].mean())} | "
                     f"{delta[:, j].mean() * 100:+.1f} | {fmt(delta[:, j].std(ddof=1))} |")
    lines.append("")
    mood = [j for j, p in enumerate(d.playlists) if re.search(r"gym|chill|happy|sad|flow|dance|sleep|focus|study|workout",
                                                             names.get(p, ""), re.I)]
    if mood:
        lines += ["Mood/activity playlists (matched by name; not selected on the result), same config:", "",
                  "| playlist | n | before → after (%) | Δ pp | SD of Δ |", "|---|---|---|---|---|"]
        for j in mood:
            lines.append(f"| {names.get(d.playlists[j], d.playlists[j])} | {int(n_pl[j])} | "
                         f"{fmt(base_h[:, j].mean())} → {fmt(best_h[:, j].mean())} | {delta[:, j].mean() * 100:+.1f} | "
                         f"{fmt(delta[:, j].std(ddof=1))} |")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    if a.out:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
