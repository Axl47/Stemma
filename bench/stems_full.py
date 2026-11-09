"""Separate the whole library into vocal and instrumental stems, and embed each.

Usage: python bench/stems_full.py <base_url> <out_dir> [--limit N]

Needs CUDA, demucs, av, soxr, onnxruntime and the engine on PYTHONPATH (AUDIO_EMBED_MODELS for the weights).
<base_url> serves list.jsonl ({key, file} per line) and audio/<file>. A file not there yet is waited for, so this
can start while the files are still being copied. Each song writes <out_dir>/stems/<key>.vocals.opus and .rest.opus
(stereo, 96 kbit/s) and then <out_dir>/vectors/<key>.npz (mix, vocals, rest, vocal_energy), which marks it done, so
a re-run resumes. "rest" is the mix minus the vocals.
"""

import argparse
import ctypes
import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import av
import numpy as np
import soxr
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from audio_embed_engine import DiscogsEffnet
from audio_embed_engine.audio import SAMPLE_RATE, log_mel, patches

OPUS_RATE = 48_000
WAIT_S = 30
MAX_WAITS = 240  # two hours for one file: the copy stopped, so skip it and let a re-run retry


def fetch(url: str, dest: Path) -> None:
    for _ in range(MAX_WAITS):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(WAIT_S)
    raise TimeoutError(url)


def load_stereo(path: Path, rate: int) -> np.ndarray:
    planar = av.AudioResampler(format="fltp", layout="stereo")
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        source = s.rate
        pcm = np.concatenate([f.to_ndarray() for fr in c.decode(s) for f in planar.resample(fr)], axis=1)
    return soxr.resample(pcm.T, source, rate, quality="HQ").astype(np.float32)  # (samples, 2)


def write_opus(path: Path, pcm: np.ndarray, rate: int) -> None:
    """Writes (samples, 2) float audio, via a temp name so a crash never leaves a partial stem."""
    audio = np.clip(soxr.resample(pcm, rate, OPUS_RATE, quality="HQ"), -1, 1).astype(np.float32)
    tmp = path.with_suffix(".part")
    with av.open(str(tmp), "w", format="ogg") as c:
        stream = c.add_stream("libopus", rate=OPUS_RATE, layout="stereo")
        stream.bit_rate = 96_000
        frame = av.AudioFrame.from_ndarray(audio.T, format="fltp", layout="stereo")
        frame.sample_rate = OPUS_RATE
        for packet in stream.encode(frame):
            c.mux(packet)
        for packet in stream.encode(None):
            c.mux(packet)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep the PC awake while this runs

    stems, vectors = args.out / "stems", args.out / "vectors"
    stems.mkdir(parents=True, exist_ok=True)
    vectors.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{args.base_url}/list.jsonl", timeout=60) as r:
        tracks = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    todo = [t for t in tracks if not (vectors / f"{t['key']}.npz").exists()][: args.limit]
    print(f"{len(todo)} to do of {len(tracks)}", flush=True)

    demucs = get_model("htdemucs").cuda().eval()
    model = DiscogsEffnet()
    effnet = model._sessions[0]

    def embed(mono: np.ndarray) -> np.ndarray:
        audio = soxr.resample(mono, demucs.samplerate, SAMPLE_RATE, quality="HQ").astype(np.float32)
        e = np.asarray(effnet.run(["embeddings"], {"melspectrogram": patches(log_mel(audio), model.PATCH, model.HOP)})[0])
        m = e.mean(0)
        return (m / max(float(np.linalg.norm(m)), 1e-12)).astype(np.float32)

    start, done = time.perf_counter(), 0
    for i, t in enumerate(todo, 1):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / t["file"]
                fetch(f"{args.base_url}/audio/{urllib.request.quote(t['file'])}", src)
                pcm = load_stereo(src, demucs.samplerate)
            wav = torch.from_numpy(pcm.T)
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std()
            with torch.no_grad():
                sources = apply_model(demucs, ((wav - mean) / std)[None], device="cuda", split=True, overlap=0.25, progress=False)[0]
            vocals = (sources[demucs.sources.index("vocals")] * std + mean).cpu().numpy().T
            rest = pcm - vocals
            write_opus(stems / f"{t['key']}.vocals.opus", vocals, demucs.samplerate)
            write_opus(stems / f"{t['key']}.rest.opus", rest, demucs.samplerate)
            mix_m, voc_m = pcm.mean(1), vocals.mean(1)
            energy = float((voc_m**2).sum() / max(float((mix_m**2).sum()), 1e-12))
            target = vectors / f"{t['key']}.npz"
            part = target.with_suffix(".part.npz")
            np.savez(part, mix=embed(mix_m), vocals=embed(voc_m), rest=embed(rest.mean(1)), vocal_energy=energy)
            part.replace(target)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {t['key']}: {type(ex).__name__}: {ex}", flush=True)
            continue
        done += 1
        if done % 10 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}  {(time.perf_counter() - start) / done:.1f} s/track", flush=True)


if __name__ == "__main__":
    main()
