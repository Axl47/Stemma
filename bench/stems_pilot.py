"""Pilot: do vocal / instrumental embeddings predict playlists better than the full mix?

Usage: python bench/stems_pilot.py run <bundle_dir> <out_dir>   (separate and embed, resumable, needs CUDA)
       python bench/stems_pilot.py eval <bundle_dir> <out_dir>

<bundle_dir> holds sample.jsonl ({key, file, primary, artists, playlists, mix} per song, `mix` being the
engine's cached full-track vector) and audio/. Stems are never written to disk: each song is separated
with Demucs htdemucs, "rest" is mix minus vocals, and all three are embedded with Discogs-EffNet into
<out_dir>/<key>.npz. Eval uses the same never-seen-artist folds as playlist_signal.py on the sample only,
averaged over repeated fold splits, and compares every representation with the mix.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from playlist_signal import auc, clean_train, folds_by_artist, score_ridge

REPS = {
    "mix": ["mix"],
    "vocals": ["vocals"],
    "rest": ["rest"],
    "mix+vocals": ["mix", "vocals"],
    "mix+rest": ["mix", "rest"],
    "mix+vocals+rest": ["mix", "vocals", "rest"],
}


def load_sample(bundle: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(bundle / "sample.jsonl", encoding="utf-8")]


def run(bundle: Path, out: Path) -> None:
    import av
    import soxr
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from audio_embed_engine import DiscogsEffnet
    from audio_embed_engine.audio import SAMPLE_RATE, log_mel, patches

    assert torch.cuda.is_available(), "CUDA is required"
    out.mkdir(parents=True, exist_ok=True)
    demucs = get_model("htdemucs").cuda().eval()
    model = DiscogsEffnet()
    effnet = model._sessions[0]

    def embed(pcm: np.ndarray) -> np.ndarray:
        audio = soxr.resample(pcm, demucs.samplerate, SAMPLE_RATE, quality="HQ").astype(np.float32)
        emb = np.asarray(effnet.run(["embeddings"], {"melspectrogram": patches(log_mel(audio), model.PATCH, model.HOP)})[0])
        m = emb.mean(0)
        return (m / max(float(np.linalg.norm(m)), 1e-12)).astype(np.float32)

    def load_stereo(path: Path) -> np.ndarray:
        planar = av.AudioResampler(format="fltp", layout="stereo")
        with av.open(str(path)) as c:
            s = c.streams.audio[0]
            rate = s.rate
            pcm = np.concatenate([f.to_ndarray() for fr in c.decode(s) for f in planar.resample(fr)], axis=1)
        return soxr.resample(pcm.T, rate, demucs.samplerate, quality="HQ").T.astype(np.float32)

    sample = load_sample(bundle)
    start, done = time.perf_counter(), 0
    for i, s in enumerate(sample, 1):
        target = out / f"{s['key']}.npz"
        if target.exists():
            continue
        try:
            wav = torch.from_numpy(load_stereo(bundle / "audio" / s["file"]))
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std()
            with torch.no_grad():
                sources = apply_model(demucs, ((wav - mean) / std)[None], device="cuda", split=True, overlap=0.25, progress=False)[0]
            sources = (sources * std + mean).cpu()
            vocals = sources[demucs.sources.index("vocals")]
            mix_mono, voc_mono = wav.mean(0).numpy(), vocals.mean(0).numpy()
            rest_mono = mix_mono - voc_mono
            ratio = float((voc_mono**2).sum() / max(float((mix_mono**2).sum()), 1e-12))
            np.savez(target, mix=embed(mix_mono), vocals=embed(voc_mono), rest=embed(rest_mono), vocal_energy=ratio)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {s['key']}: {type(ex).__name__}: {ex}", flush=True)
            continue
        done += 1
        print(f"{i}/{len(sample)}  {(time.perf_counter() - start) / done:.1f} s/track", flush=True)


def evaluate(bundle: Path, out: Path, repeats: int = 20) -> None:
    rows = [(s, np.load(out / f"{s['key']}.npz")) for s in load_sample(bundle) if (out / f"{s['key']}.npz").exists()]
    names = sorted({p for s, _ in rows for p in s["playlists"]})
    y = np.array([[n in s["playlists"] for n in names] for s, _ in rows], bool)
    artists = [frozenset(s["artists"]) for s, _ in rows]
    primary = [s["primary"] for s, _ in rows]
    sanity = np.array([float(np.dot(r["mix"], np.asarray(s["mix"], np.float32))) for s, r in rows])
    print(f"{len(rows)} songs, {len(names)} playlists; cosine of pilot mix vs engine's cached full vector: "
          f"median {np.median(sanity):.4f}, min {sanity.min():.4f}")
    print(f"vocal energy share: median {np.median([float(r['vocal_energy']) for _, r in rows]):.3f}")

    aucs: dict[str, np.ndarray] = {}
    idx = np.arange(len(rows))
    for rep, parts in REPS.items():
        x = np.concatenate([np.stack([r[p] for _, r in rows]) for p in parts], axis=1) / np.sqrt(len(parts))
        per_seed = []
        for seed in range(repeats):
            oof, seen = np.zeros(y.shape), np.zeros(y.shape, bool)
            for test in folds_by_artist(idx, primary, 5, seed):
                train = clean_train(np.setdiff1d(idx, test), test, artists)
                oof[test] = score_ridge(x[test], x[train], y[train].astype(float))
                seen[test] = y[train].any(0)
            per_seed.append([auc(oof[seen[:, p], p], y[seen[:, p], p]) for p in range(len(names))])
        aucs[rep] = np.nanmean(per_seed, axis=0)
        # one number per fold split, so the spread reflects how the split lands
        aucs[f"{rep}/seeds"] = np.nanmean(per_seed, axis=1)

    base = aucs["mix/seeds"]
    print(f"{'representation':<18} {'mean AUC':>8} {'minus mix':>10} {'seeds better':>13}")
    for rep in REPS:
        s = aucs[f"{rep}/seeds"]
        print(f"{rep:<18} {s.mean():>8.3f} {(s - base).mean():>+10.3f} {int((s > base).sum()):>10}/{repeats}")
    print("\nper playlist AUC (songs):", " ".join(f"| {r}" for r in REPS))
    for p, n in enumerate(names):
        print(f"{n} ({int(y[:, p].sum())}) " + " ".join(f"| {aucs[r][p]:.3f}" for r in REPS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    (run if a.cmd == "run" else evaluate)(a.bundle, a.out)


if __name__ == "__main__":
    main()
