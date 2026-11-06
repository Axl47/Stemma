"""Does embedding the full track beat the 30 s Deezer preview? Same model, same tracks, three inputs.

Usage:
  python bench/full_vs_preview.py <previews.npz> <yt_dir> <sample-tracks.jsonl> <sample-labels.jsonl> \
      <fixture-tracks.jsonl> [--out bench/full_vs_preview.json]

Inputs: previews.npz from bench/fetch_audio.py, yt_dir from bench/fetch_youtube.py. Only tracks with
both a preview and a YouTube download are scored, so every variant sees the same tracks:
- preview:  the Deezer 30 s preview (what the engine embeds today)
- full:     the whole YouTube Music track
- center30: the middle 30 s of the YouTube track. It separates "more audio" from "different source/codec".
Pair AUC comes with a paired bootstrap over tracks for the full-minus-preview difference.
"""

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import soxr
from metrics import pair_auc, playlist_ranking, style_knn

from audio_embed_engine.audio import SAMPLE_RATE, decode
from audio_embed_engine.models import DiscogsEffnet


def decode_file(path: Path) -> np.ndarray:
    """Any ffmpeg-readable file -> mono float32 at 16 kHz, resampled with soxr like the engine's decode."""
    planar = av.AudioResampler(format="fltp")  # keeps rate and layout; to_ndarray() -> (channels, samples)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        chunks = [f.to_ndarray() for frame in container.decode(stream) for f in planar.resample(frame)]
    pcm = np.concatenate(chunks, axis=1).mean(axis=0)
    return soxr.resample(pcm, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)


def center(audio: np.ndarray, seconds: float = 30.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    start = max((len(audio) - n) // 2, 0)
    return audio[start : start + n]


def bootstrap_delta(keys, a, b, labels, artists, n=300, seed=0) -> tuple[float, float, float]:
    """Pair-AUC(b) - pair-AUC(a), resampling tracks with replacement: (mean, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.choice(len(keys), len(keys))
        # duplicate picks get distinct keys so pair_auc treats them as separate rows
        ks = [f"{keys[i]}#{j}" for j, i in enumerate(idx)]
        lab = {k: labels[keys[i]] for k, i in zip(ks, idx)}
        art = {k: artists[keys[i]] | {f"self{i}"} for k, i in zip(ks, idx)}  # a track never pairs with its copy
        deltas.append(pair_auc(ks, b[idx], lab, art, True) - pair_auc(ks, a[idx], lab, art, True))
    d = np.array(deltas)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("previews")
    ap.add_argument("yt_dir", type=Path)
    ap.add_argument("tracks")
    ap.add_argument("labels")
    ap.add_argument("fixture")
    ap.add_argument("--out", default="bench/full_vs_preview.json")
    args = ap.parse_args()

    with np.load(args.previews) as z:
        previews = {k: z[k].tobytes() for k in z.files}
    downloads = {
        d["key"]: args.yt_dir / d["file"]
        for line in (args.yt_dir / "manifest.jsonl").open()
        if (d := json.loads(line))["status"] == "ok"
    }
    keys = sorted(previews.keys() & downloads.keys())
    print(f"{len(previews)} previews, {len(downloads)} downloads, {len(keys)} in both")

    model = DiscogsEffnet()
    audio = {"preview": [decode(previews[k]) for k in keys]}
    audio["full"] = [decode_file(downloads[k]) for k in keys]
    audio["center30"] = [center(a) for a in audio["full"]]
    emb: dict[str, np.ndarray] = {}
    infer_ms: dict[str, float] = {}
    for name, audios in audio.items():
        model.embed(audios[:1])  # warm-up
        t = time.perf_counter()
        emb[name] = np.stack([e.vector for i in range(0, len(audios), 16) for e in model.embed(audios[i : i + 16])])
        infer_ms[name] = 1000 * (time.perf_counter() - t) / len(keys)

    tracks = {(d := json.loads(l)).get("isrc") or d["key"]: d for l in open(args.tracks)}
    playlists = {d["key"]: d["playlists"] for l in open(args.labels) for d in [json.loads(l)]}
    fixture = {(d := json.loads(l))["isrc"]: d for l in open(args.fixture)}
    s_idx = [i for i, k in enumerate(keys) if k in tracks]
    s_keys = [keys[i] for i in s_idx]
    labels = {k: set(playlists[tracks[k]["key"]]) for k in s_keys}
    artists = {k: set(tracks[k]["artists"]) for k in s_keys}
    f_idx = [i for i, k in enumerate(keys) if k in fixture]
    f_keys = [keys[i] for i in f_idx]

    results: dict[str, object] = {"tracks_scored": {"sample": len(s_keys), "fixture": len(f_keys)}}
    for name, e in emb.items():
        r = {
            "infer_ms_per_track": infer_ms[name],
            "pair_auc_no_shared_artist": pair_auc(s_keys, e[s_idx], labels, artists, exclude_shared_artist=True),
            **playlist_ranking(s_keys, e[s_idx], labels, artists),
            **style_knn(
                f_keys,
                e[f_idx],
                {k: fixture[k]["style"] for k in f_keys},
                {k: fixture[k]["artistLabel"] for k in f_keys},
            ),
        }
        results[name] = r
        print(name, json.dumps({k: round(v, 3) for k, v in r.items()}))

    for name in ("full", "center30"):
        mean, lo, hi = bootstrap_delta(s_keys, emb["preview"][s_idx], emb[name][s_idx], labels, artists)
        results[f"pair_auc_delta_{name}_vs_preview"] = {"mean": mean, "ci95": [lo, hi]}
        print(f"pair AUC {name} - preview: {mean:+.3f} (95% CI {lo:+.3f} .. {hi:+.3f})")

    cos = np.sum(emb["preview"] * emb["full"], axis=1)
    order = np.argsort(cos)
    results["preview_full_cosine"] = {"median": float(np.median(cos)), "p10": float(np.percentile(cos, 10))}
    results["least_similar"] = [{"key": keys[i], "cos": float(cos[i])} for i in order[:8]]
    print(f"cos(preview, full): median {np.median(cos):.3f}, p10 {np.percentile(cos, 10):.3f}")
    for i in order[:8]:
        t = tracks.get(keys[i]) or fixture[keys[i]]
        print(f"  {cos[i]:.3f}  {t['artists'][0]} - {t['title']}")
    Path(args.out).write_text(json.dumps(results, indent=1))
    np.savez(Path(args.out).with_suffix(".npz"), keys=np.array(keys), **emb)  # for ad-hoc follow-ups


if __name__ == "__main__":
    main()
