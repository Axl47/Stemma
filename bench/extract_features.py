# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["beat-this==1.1.0", "essentia==2.1b6.dev1389", "numpy", "soxr"]
# [tool.uv]
# exclude-newer = "2026-11-12T00:00:00Z"
# ///
"""Per-song features for ordering songs by how they hand off: tempo and beats, musical key,
loudness over time, and when vocals are present. One JSON line per song.

Usage: uv run --no-project --python 3.12 bench/extract_features.py <sources.jsonl> <out.jsonl>
           [--keys FILE] [--limit N] [--batch N] [--threads N]
           [--device mps|cpu|both] [--timings] [--compare-rest] [--src RSYNC_SRC]
       uv run --no-project --python 3.12 bench/extract_features.py report <sources.jsonl> <out.jsonl>

The script carries its own dependencies (PEP 723), so it never touches the project's environment.
Audio stays on the storage box: each batch of full mixes and vocals stems is rsynced to a temp dir
(the next batch downloads while this one is processed) and deleted afterwards, as in embed_full.py.
`sources.jsonl` is the box's data/sources.jsonl; `--keys` restricts the run to the keys listed in a file
(one per line). Resumable: songs already in <out.jsonl> are skipped, a half-written last line is cut off,
and songs that failed are retried. Details and the output schema: bench/extract_features.md.
"""

from __future__ import annotations

import os
import sys


def _threads_from_argv() -> int:
    """Read --threads before numpy and torch load: their thread pools are sized at import."""
    argv = sys.argv
    return int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 2


THREADS = _threads_from_argv()
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(THREADS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops mps lacks run on the CPU instead of failing

import argparse
import json
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import essentia.standard as es
import numpy as np
import soxr
import torch

SR = 44100  # decode rate: Essentia's default, and plenty for loudness
BEAT_SR = 22050  # beat_this input rate
VOCAL_SR = 16000
FETCH_TIMEOUT_S = 180
STALL_TIMEOUT_S = 15
FETCH_ATTEMPTS = 5
EDGE_S = 30.0  # "first 30 s" / "last 30 s" windows
SILENCE_LUFS = -70.0  # ebur128's absolute gate; also what we call silence
MIN_TAIL_S = 3.0  # a shorter ending is just an ending, not a fade-out or quiet tail

# --- tempo -------------------------------------------------------------------------------------


def span_intervals(beats: np.ndarray, k: int) -> np.ndarray:
    """Mean beat interval over every k-beat span. Beat times sit on a 20 ms grid (the model's frame
    rate), so one interval is off by up to 4% at 120 BPM; averaging over k beats divides that by k."""
    return (beats[k:] - beats[:-k]) / k if len(beats) > k else np.empty(0)


def bpm_of(beats: np.ndarray, k: int = 8) -> float | None:
    """Tempo as 60 / the median k-beat-span interval; None when there are too few beats."""
    spans = span_intervals(beats, k)
    return round(60 / float(np.median(spans)), 1) if len(spans) >= 3 else None


def tempo_features(beats: np.ndarray, downbeats: np.ndarray, dur: float) -> dict[str, Any]:
    """bpm: whole song. bpm_first/bpm_last: beats in the first/last 30 s. bpm_fold: bpm doubled or halved
    into [80, 160), where most songs are felt; bpm_ambig: true when that moved it or it sits within 5 of an
    edge, i.e. half or double time is plausible. bpb: beats per bar (median). reg: share of 4-beat spans
    within 3% of the median tempo (1 = metronome, low = rubato, tempo changes or a poor track)."""
    bpm = bpm_of(beats)
    fold, ambig, reg = None, None, None
    if bpm:
        fold = bpm
        while fold < 80:
            fold *= 2
        while fold >= 160:
            fold /= 2
        ambig = fold != bpm or fold < 85 or fold >= 150
        fold = round(fold, 1)
        reg = round(
            float(np.mean(np.abs(span_intervals(beats, 4) / np.median(span_intervals(beats, 8)) - 1) < 0.03)), 3
        )
    bpb = None
    if len(downbeats) >= 3:
        idx = np.searchsorted(beats, downbeats - 0.011)  # downbeats are beats: find each one's index
        bpb = int(np.median(np.diff(idx)))
    return {
        "bpm": bpm,
        "bpm_first": bpm_of(beats[beats <= EDGE_S]),
        "bpm_last": bpm_of(beats[beats >= dur - EDGE_S]),
        "bpm_fold": fold,
        "bpm_ambig": ambig,
        "bpb": bpb,
        "reg": reg,
        "beats": [round(float(t), 2) for t in beats],
        "downbeats": [round(float(t), 2) for t in downbeats],
    }


# --- musical key -------------------------------------------------------------------------------

PITCHES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
ENHARMONIC = {"C#": "Db", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}
CAMELOT_NUMBER = {
    "B": 1,
    "F#": 2,
    "Db": 3,
    "Ab": 4,
    "Eb": 5,
    "Bb": 6,
    "F": 7,
    "C": 8,
    "G": 9,
    "D": 10,
    "A": 11,
    "E": 12,
}


def pitch_class(name: str) -> int:
    return PITCHES.index(ENHARMONIC.get(name, name))


def camelot(tonic: str, mode: str) -> str:
    """Camelot wheel code (8B = C major, 8A = A minor): a key's relative major/minor share the number."""
    major = PITCHES[(pitch_class(tonic) + 3) % 12] if mode == "minor" else tonic
    return f"{CAMELOT_NUMBER[ENHARMONIC.get(major, major)]}{'A' if mode == 'minor' else 'B'}"


def key_of(audio: np.ndarray) -> list[Any]:
    """[tonic, mode, strength] from Essentia's KeyExtractor with default settings (bgate profile)."""
    tonic, mode, strength = es.KeyExtractor()(audio)
    return [ENHARMONIC.get(tonic, tonic), mode, round(float(strength), 3)]


def key_relation(a: list[Any], b: list[Any]) -> str:
    """How key b relates to key a: exact, fifth (same mode, tonics a fifth apart), relative (C major /
    A minor), parallel (same tonic, other mode), or other."""
    (ta, ma), (tb, mb) = (pitch_class(a[0]), a[1]), (pitch_class(b[0]), b[1])
    if (ta, ma) == (tb, mb):
        return "exact"
    if ma == mb and (ta - tb) % 12 in (5, 7):
        return "fifth"
    if ma != mb and ((ta + 3) % 12 == tb if ma == "minor" else (tb + 3) % 12 == ta):
        return "relative"
    return "parallel" if ta == tb else "other"


# --- loudness ----------------------------------------------------------------------------------

FRAME_RE = re.compile(r"t:\s*([\d.]+)\s+TARGET:\S+ LUFS\s+M:\s*(-?[\d.]+|-inf|nan)\s+S:\s*(-?[\d.]+|-inf|nan)")
INTEGRATED_RE = re.compile(r"Integrated loudness:\s+I:\s+(-?[\d.]+) LUFS")
LRA_RE = re.compile(r"Loudness range:\s+LRA:\s+([\d.]+) LU")


def run_ebur128(pcm: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """ffmpeg's ebur128 on stereo float PCM piped to stdin: (integrated LUFS, LRA, M per 0.1 s, S per 0.1 s).
    The 0.1 s series starts at t = 0.1 s. S is undefined (-120.7) for the first 3 s."""
    cmd = ["ffmpeg", "-nostats", "-v", "info", "-threads", str(THREADS), "-f", "f32le", "-ar", str(SR), "-ac", "2"]
    cmd += ["-i", "pipe:0", "-filter_complex", "ebur128=peak=none", "-f", "null", "-"]
    err = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, check=True).stderr.decode(errors="replace")
    frames = FRAME_RE.findall(err)
    integrated, lra = INTEGRATED_RE.search(err), LRA_RE.search(err)
    if not (frames and integrated and lra):
        raise RuntimeError("could not parse ebur128 output")

    def series(column: int) -> np.ndarray:
        return np.array([-120.7 if f[column] in ("-inf", "nan") else float(f[column]) for f in frames])

    return float(integrated[1]), float(lra[1]), series(1), series(2)


def energy_mean(db: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average a loudness series in the energy domain (what ebur128 itself does), in dB."""
    return 10 * np.log10(np.mean(10 ** (np.maximum(db, -120.0) / 10), axis=axis) + 1e-12)


def loudness_features(pcm: np.ndarray) -> dict[str, Any]:
    """lufs/lra: integrated loudness and loudness range of the song. st: short-term loudness once per second,
    st[k] at t = k+1 s (the first 2 values use the audio elapsed so far: S is undefined for 3 s).
    typical: the median second among those within 10 LU of lufs. level_t: when the song first reaches near
    that level, i.e. the first second from which 3 s of loudness stay within 3 LU of it: 0-1 is a cold start,
    later a build. end: where the audio really ends (trailing digital silence cut off).
    tail: what follows the last second within 3 LU of typical, up to `end`, as {start, len, drop}; null when
    shorter than MIN_TAIL_S. drop is how far the last audible second sits below typical: a long len with a
    large drop is a fade-out, a long len with a small drop a soft outro. The per-second levels come from the
    400 ms momentary values and the 3 s windows look forward, so, unlike the standard short-term value,
    they do not lag and the times stay honest."""
    lufs, lra, m, s = run_ebur128(pcm)
    n = len(m) // 10  # whole seconds
    if n < 6:
        raise RuntimeError(f"too short for loudness ({len(m) / 10:.1f} s)")
    sec = energy_mean(m[: n * 10].reshape(n, 10))
    st = np.maximum(s[9 : n * 10 : 10], SILENCE_LUFS)
    for k in range(min(2, n)):
        st[k] = max(float(energy_mean(m[: 10 * (k + 1)])), SILENCE_LUFS)
    loud = sec[sec >= lufs - 10]
    typical = float(np.median(loud)) if len(loud) else lufs
    audible = np.flatnonzero(m > -60)
    end = float(audible[-1] + 1) / 10 if len(audible) else 0.0
    win = energy_mean(np.lib.stride_tricks.sliding_window_view(sec, 3), axis=1)
    near = np.flatnonzero(win >= typical - 3)
    last_near = np.flatnonzero(sec >= typical - 3)
    tail = None
    if len(last_near) and end - (start := float(last_near[-1] + 1)) >= MIN_TAIL_S:
        last_second = float(sec[min(int(end), n) - 1])
        tail = {"start": round(start, 1), "len": round(end - start, 1), "drop": round(typical - last_second, 1)}
    return {
        "lufs": round(lufs, 1),
        "lra": round(lra, 1),
        "st": [round(float(v), 1) for v in st],
        "typical": round(typical, 1),
        "level_t": float(near[0]) if len(near) else None,
        "end": round(end, 1),
        "tail": tail,
    }


# --- vocals ------------------------------------------------------------------------------------

FRAME_S = 0.05
# A frame has vocals when the vocals stem is within ON_REL dB of the full mix in that frame (and not near
# silent): demucs leaves faint instrument bleed in the stem, which fixed dBFS thresholds mistake for singing
# in quiet songs. Hysteresis: a segment ends below OFF_REL.
ON_REL, OFF_REL = -20.0, -26.0
ON_FLOOR_DB, OFF_FLOOR_DB = -42.0, -48.0  # dBFS RMS of the stem
MIN_SEGMENT_S, MIN_GAP_S = 1.0, 1.5  # drop shorter vocal blips; bridge shorter pauses (breaths, word gaps)
# A segment under ISOLATED_MAX_S with no other vocals within ISOLATED_GAP_S either side is a stray chop or
# bleed, not singing: it must not set the first or last vocal time.
ISOLATED_MAX_S, ISOLATED_GAP_S = 3.0, 10.0
INSTRUMENTAL_ENERGY = 0.02  # below this share of the mix's energy in the stem, call the song instrumental


def frame_db(x: np.ndarray) -> np.ndarray:
    n = int(FRAME_S * VOCAL_SR)
    frames = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)


def vocal_segments(stem: np.ndarray, mix: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) seconds where the vocals stem is audible next to the mix (both 16 kHz mono): short-frame
    RMS with on/off thresholds, pauses under MIN_GAP_S merged, then segments under MIN_SEGMENT_S dropped."""
    n = min(len(stem), len(mix))
    stem_db, rel = frame_db(stem[:n]), frame_db(stem[:n]) - frame_db(mix[:n])
    segs: list[list[int]] = []
    on = False
    for i, (v, r) in enumerate(zip(stem_db, rel, strict=True)):
        if not on and r > ON_REL and v > ON_FLOOR_DB:
            segs.append([i, i + 1])
            on = True
        elif on and (r < OFF_REL or v < OFF_FLOOR_DB):
            on = False
        elif on:
            segs[-1][1] = i + 1
    merged: list[list[int]] = []
    for seg in segs:
        if merged and (seg[0] - merged[-1][1]) * FRAME_S < MIN_GAP_S:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    segments = [(a * FRAME_S, b * FRAME_S) for a, b in merged if (b - a) * FRAME_S >= MIN_SEGMENT_S]
    far = float("inf")

    def isolated(i: int) -> bool:
        start, end = segments[i]
        before = start - segments[i - 1][1] if i > 0 else far
        after = segments[i + 1][0] - end if i + 1 < len(segments) else far
        return end - start < ISOLATED_MAX_S and min(before, after) >= ISOLATED_GAP_S

    return [seg for i, seg in enumerate(segments) if len(segments) == 1 or not isolated(i)]


def vocal_features(stem: np.ndarray, mix: np.ndarray, dur: float) -> dict[str, Any]:
    """first/last: when vocals first start and last stop. share: fraction of the song with vocals. energy: the
    stem's share of the mix's energy; under INSTRUMENTAL_ENERGY the song counts as instrumental (first and
    last null, share 0), as the stem then holds only bleed or the odd vocal chop."""
    n = min(len(stem), len(mix))
    energy = float(np.sum(stem[:n].astype(np.float64) ** 2) / (np.sum(mix[:n].astype(np.float64) ** 2) + 1e-12))
    segs = vocal_segments(stem, mix) if energy >= INSTRUMENTAL_ENERGY else []
    if not segs:
        return {"first": None, "last": None, "share": 0.0, "energy": round(energy, 3)}
    return {
        "first": round(segs[0][0], 1),
        "last": round(segs[-1][1], 1),
        "share": round(min(sum(b - a for a, b in segs) / dur, 1.0), 3),
        "energy": round(energy, 3),
    }


# --- audio, beats ------------------------------------------------------------------------------


def decode(path: Path, rate: int, channels: int) -> np.ndarray:
    """Any file ffmpeg reads -> float32 samples, shape (n,) for mono or (n, channels)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", str(THREADS), "-i", str(path), "-vn"]
    cmd += ["-ac", str(channels), "-ar", str(rate), "-f", "f32le", "pipe:1"]
    pcm = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
    return pcm.reshape(-1, channels) if channels > 1 else pcm


class BeatTrackers:
    """beat_this `final0` without the DBN post-processor. `mps` runs on the Apple GPU with a CPU fallback:
    if mps fails once it is dropped for the rest of the run. `both` runs every song on both, to time them
    (the first one that works supplies the beats)."""

    def __init__(self, device: str) -> None:
        from beat_this.inference import Audio2Beats

        torch.set_num_threads(THREADS)
        self.compare = device == "both"
        self.trackers: dict[str, Any] = {}
        for name in ("mps", "cpu") if device in ("mps", "both") else ("cpu",):
            try:
                self.trackers[name] = Audio2Beats(checkpoint_path="final0", device=name, dbn=False)
            except Exception as e:  # noqa: BLE001
                print(f"beat_this on {name} unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        if not self.trackers:
            raise SystemExit("no beat tracker could be loaded")

    def run(self, mono22k: np.ndarray, timings: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(beats, downbeats) in seconds; fills timings['beats_<device>']."""
        result = None
        results: dict[str, np.ndarray] = {}
        for name in list(self.trackers):
            t0 = time.perf_counter()
            try:
                beats, downbeats = self.trackers[name](mono22k, BEAT_SR)
                if name == "mps":
                    torch.mps.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"beat_this on {name} failed ({type(e).__name__}: {e}); dropping it", file=sys.stderr)
                del self.trackers[name]
                continue
            timings[f"beats_{name}"] = round(time.perf_counter() - t0, 2)
            result = result or (np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float))
            results[name] = np.asarray(beats, dtype=float)
            if not self.compare:
                break
        if len(results) == 2:  # share of cpu beats that mps also found within 25 ms (and the reverse): 1.0 = same
            a, b = results["mps"], results["cpu"]
            near = lambda x, y: float(np.mean([np.min(np.abs(y - v)) < 0.025 for v in x])) if len(x) else 1.0
            timings["beats_match"] = round(min(near(a, b), near(b, a)), 3)
        if result is None:
            raise RuntimeError("no beat tracker worked")
        return result


# --- one song ----------------------------------------------------------------------------------


def timed[T](timings: dict[str, float], stage: str, fn: Callable[..., T], *args: Any) -> T:
    t0 = time.perf_counter()
    result = fn(*args)
    timings[stage] = round(time.perf_counter() - t0, 2)
    return result


def extract(key: str, root: Path, mix: str, trackers: BeatTrackers, compare_rest: bool) -> dict[str, Any]:
    """All features for one song whose files sit under root (tracks/, stems/)."""
    t: dict[str, float] = {}
    stereo = timed(t, "decode", decode, root / "tracks" / mix, SR, 2)
    mono = stereo.mean(axis=1, dtype=np.float32)
    dur = len(mono) / SR
    mono22k = timed(t, "resample", lambda: soxr.resample(mono, SR, BEAT_SR))
    beats, downbeats = trackers.run(mono22k, t)
    if len(beats) < 4:
        raise RuntimeError(f"only {len(beats)} beats found")
    edge = int(EDGE_S * SR)
    all_key = timed(t, "key", key_of, mono)
    tonal = {"all": all_key, "first": key_of(mono[:edge]), "last": key_of(mono[-edge:])}
    tonal["camelot"] = camelot(all_key[0], all_key[1])
    if compare_rest:
        rest = timed(t, "rest_decode", decode, root / "stems" / f"{key}.rest.opus", SR, 1)
        tonal["rest"] = timed(t, "rest_key", key_of, rest)
    loud = timed(t, "loudness", loudness_features, stereo)
    stem = timed(t, "vocals_decode", decode, root / "stems" / f"{key}.vocals.opus", VOCAL_SR, 1)
    voc = timed(t, "vocals", vocal_features, stem, soxr.resample(mono, SR, VOCAL_SR), dur)
    return {
        "key": key,
        "dur": round(dur, 1),
        "tempo": tempo_features(beats, downbeats, dur),
        "tonal": tonal,
        "loud": loud,
        "voc": voc,
        "t": t,
    }


# --- batches, resume ---------------------------------------------------------------------------


def fetch(files: list[str], src: str, dest: Path) -> float:
    """rsync a batch into dest on a fresh connection per attempt (a long-lived one can stall without ever
    failing); returns seconds. rsync exit 23 (some file missing on the box) is accepted: extract() fails
    those songs alone."""
    t0 = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "list").write_text("\n".join(files))
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            # --timeout aborts a connection that has gone quiet (seen: ~12 KB/s for minutes, where a new
            # connection takes seconds); the overall timeout only bounds a batch that keeps trickling
            cmd = ["rsync", "-t", f"--timeout={STALL_TIMEOUT_S}", "--files-from", str(dest / "list"), src, str(dest)]
            code = subprocess.run(cmd, check=False, timeout=FETCH_TIMEOUT_S).returncode
            if code in (0, 23):
                return time.perf_counter() - t0
            raise subprocess.CalledProcessError(code, "rsync")
        except (subprocess.SubprocessError, OSError) as e:
            print(f"fetch attempt {attempt}/{FETCH_ATTEMPTS} failed: {e}", file=sys.stderr)
            if attempt == FETCH_ATTEMPTS:
                raise
            time.sleep(5)
    raise AssertionError("unreachable")


def read_done(out: Path) -> set[str]:
    """Keys already extracted. Cuts a half-written last line (stopped mid-write); error lines don't count."""
    if not out.exists():
        return set()
    data = out.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        out.write_bytes(data)
    lines = (json.loads(line) for line in data.decode().splitlines())
    return {d["key"] for d in lines if "error" not in d}


def stop(*_: object) -> None:
    raise KeyboardInterrupt


# --- report ------------------------------------------------------------------------------------


def sparkline(values: list[float], lo: float, hi: float, width: int = 48) -> str:
    """Loudness curve as block characters, resampled to `width` characters (max of each slice)."""
    edges = np.linspace(0, len(values), width + 1).astype(int)
    cells = [max(values[a : max(b, a + 1)]) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return "".join("▁▂▃▄▅▆▇█"[int(np.clip((v - lo) / (hi - lo), 0, 0.999) * 8)] for v in cells)


def tempo_agreement(a: float | None, b: float | None) -> str:
    """Do two tempo estimates agree: same (within 2%), octave (double/half), or differ."""
    if not a or not b:
        return "-"
    ratio = max(a, b) / min(a, b)
    return "same" if ratio < 1.02 else "octave" if abs(ratio - 2) < 0.04 else "differ"


def report(sources: Path, out: Path) -> None:
    """Print one table per aspect (tempo, key, loudness, vocals) for a spot check, then agreement rates."""
    names = {
        d["key"]: f"{', '.join(d.get('artists', []))[:16]} - {d.get('title', '')}"[:34]
        for d in map(json.loads, open(sources))
    }
    recs = {d["key"]: d for d in map(json.loads, open(out)) if "error" not in d}
    print(f"{len(recs)} songs\n")
    rows: list[tuple[str, dict[str, Any]]] = [(names.get(k, k), d) for k, d in recs.items()]

    print(
        f"{'TEMPO':34} {'dur':>5} {'bpm':>6} {'first':>6} {'last':>6} {'fold':>6} {'amb':>3} {'bpb':>3} {'reg':>5} {'first~last':>10}"
    )
    for name, d in rows:
        t = d["tempo"]
        agree = tempo_agreement(t["bpm_first"], t["bpm_last"])
        print(
            f"{name:34} {d['dur']:5.0f} {t['bpm'] or '-':>6} {t['bpm_first'] or '-':>6} {t['bpm_last'] or '-':>6} {t['bpm_fold'] or '-':>6} {'y' if t['bpm_ambig'] else '':>3} {t['bpb'] or '-':>3} {t['reg'] if t['reg'] is not None else '-':>5} {agree:>10}"
        )

    has_rest = any("rest" in d["tonal"] for _, d in rows)
    print(
        f"\n{'KEY':34} {'mix':>10} {'str':>4} {'first':>9} {'last':>9} {'cam':>3}"
        + (f" {'rest':>10} {'mix~rest':>9}" if has_rest else "")
    )
    relations: list[str] = []
    for name, d in rows:
        k = d["tonal"]
        fmt = lambda x: f"{x[0]} {x[1][:3]}"
        line = (
            f"{name:34} {fmt(k['all']):>10} {k['all'][2]:4} {fmt(k['first']):>9} {fmt(k['last']):>9} {k['camelot']:>3}"
        )
        if "rest" in k:
            rel = key_relation(k["all"], k["rest"])
            relations.append(rel)
            line += f" {fmt(k['rest']):>10} {rel:>9}"
        print(line)

    print(
        f"\n{'LOUDNESS':34} {'LUFS':>5} {'LRA':>4} {'level_t':>7} {'end':>6} {'tail start+len':>15} {'drop':>5}  whole song (LUFS-20..+3)                         last 30 s (typical-40..+3)"
    )
    for name, d in rows:
        lo = d["loud"]
        tail = lo["tail"]
        tail_s = f"{tail['start']:.0f}+{tail['len']:.0f}s" if tail else "-"
        whole = sparkline(lo["st"], lo["lufs"] - 20, lo["lufs"] + 3)
        ending = sparkline(lo["st"][-30:], lo["typical"] - 40, lo["typical"] + 3, 30)
        print(
            f"{name:34} {lo['lufs']:5} {lo['lra']:4} {lo['level_t']:7} {lo['end']:6} {tail_s:>15} {tail['drop'] if tail else '':>5}  {whole}  {ending}"
        )

    print(f"\n{'VOCALS':34} {'first':>6} {'last':>6} {'share':>5} {'energy':>6}")
    for name, d in rows:
        v = d["voc"]
        print(
            f"{name:34} {v['first'] if v['first'] is not None else '-':>6} {v['last'] if v['last'] is not None else '-':>6} {v['share']:5} {v['energy']:6}"
        )

    agree = [tempo_agreement(d["tempo"]["bpm_first"], d["tempo"]["bpm_last"]) for _, d in rows]
    print(f"\ntempo first vs last 30 s: {', '.join(f'{a} {agree.count(a)}' for a in dict.fromkeys(agree))}")
    if relations:
        counts = {r: relations.count(r) for r in ("exact", "relative", "fifth", "parallel", "other")}
        near = counts["relative"] + counts["fifth"]
        print(
            f"key mix vs rest ({len(relations)} songs): {counts}; exact {counts['exact'] / len(relations):.0%}, exact+relative+fifth {(counts['exact'] + near) / len(relations):.0%}"
        )
    timing_keys = sorted({k for _, d in rows for k in d.get("t", {})})
    if timing_keys:
        print("\nper-song seconds per stage (mean, max):")
        for k in timing_keys:
            vals = [d["t"][k] for _, d in rows if k in d["t"]]
            print(f"  {k:14} {np.mean(vals):6.2f} {np.max(vals):6.2f}")


def main() -> None:
    if sys.argv[1:2] == ["report"]:
        return report(Path(sys.argv[2]), Path(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", type=Path, help="data/sources.jsonl from the box: {key, status, file}")
    ap.add_argument("out", type=Path)
    ap.add_argument("--keys", type=Path, help="only these song keys, one per line")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8, help="songs per rsync")
    ap.add_argument("--threads", type=int, default=THREADS, help="CPU threads for torch, numpy and ffmpeg")
    ap.add_argument("--device", choices=["mps", "cpu", "both"], default="mps", help="beat_this device; both = timing")
    ap.add_argument("--timings", action="store_true", help="add per-stage seconds `t` to every line")
    ap.add_argument("--compare-rest", action="store_true", help="also key the `rest` stem (pilot check)")
    ap.add_argument("--src", default="audio-box:./")
    args = ap.parse_args()

    # latest row per key wins, as in embed_full.py
    latest = {d["key"]: d.get("file") for line in open(args.sources) for d in [json.loads(line)]}
    files = {k: f for k, f in latest.items() if f}
    wanted = set(args.keys.read_text().split()) if args.keys else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(args.out)
    todo = sorted(k for k in files if (wanted is None or k in wanted) and k not in done)[: args.limit]
    print(f"{len(todo)} songs to do, {len(done)} done", file=sys.stderr)
    if not todo:
        return

    trackers = BeatTrackers(args.device)
    scratch = args.out.parent / f".tmp-{args.out.stem}"
    shutil.rmtree(scratch, ignore_errors=True)  # left over from a stopped run
    signal.signal(signal.SIGTERM, stop)  # clean up when killed, not only on ctrl-C

    def batch_files(keys: list[str]) -> list[str]:
        names = [f"tracks/{files[k]}" for k in keys] + [f"stems/{k}.vocals.opus" for k in keys]
        return names + ([f"stems/{k}.rest.opus" for k in keys] if args.compare_rest else [])

    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    start, finished, failed = time.perf_counter(), 0, 0
    with ThreadPoolExecutor(1) as pool, open(args.out, "a") as out:
        pending: Future[float] | None = pool.submit(fetch, batch_files(batches[0]), args.src, scratch / "0")
        try:
            for b, keys in enumerate(batches):
                assert pending is not None
                fetch_s = pending.result()
                last = b + 1 == len(batches)
                pending = (
                    None if last else pool.submit(fetch, batch_files(batches[b + 1]), args.src, scratch / str(b + 1))
                )
                for key in keys:
                    try:
                        rec = extract(key, scratch / str(b), files[key], trackers, args.compare_rest)
                        rec["t"]["fetch"] = round(fetch_s / len(keys), 2)
                        if not args.timings:
                            del rec["t"]
                    except Exception as e:  # noqa: BLE001
                        rec = {"key": key, "error": f"{type(e).__name__}: {e}"}
                        failed += 1
                        print(f"{key}: {rec['error']}", file=sys.stderr)
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    finished += 1
                shutil.rmtree(scratch / str(b), ignore_errors=True)
                rate = (time.perf_counter() - start) / finished
                print(
                    f"{finished}/{len(todo)}  {rate:.1f} s/song  ~{(len(todo) - finished) * rate / 60:.0f} min left  {failed} failed",
                    file=sys.stderr,
                )
        finally:
            if pending:
                pending.cancel()
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("stopped; rerun the same command to resume")
