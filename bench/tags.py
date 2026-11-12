# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["essentia==2.1b6.dev1389", "numpy", "onnx", "onnxruntime", "soxr"]
# [tool.uv]
# exclude-newer = "2026-11-12T00:00:00Z"
# ///
"""Mood, genre, danceability and voice tags for the whole library, from the stored Discogs-EffNet
vectors: Essentia's pretrained classification heads run on each ~2 s patch embedding and are averaged
per song (the Essentia convention). No audio is decoded, except in `verify`.

Usage: uv run --no-project --python 3.12 bench/tags.py run
       uv run --no-project --python 3.12 bench/tags.py verify [--n 8] [--src RSYNC_SRC]
       uv run --no-project --python 3.12 bench/tags.py report <cento_data_dir>

`run` writes ~/.cache/audio-embed-engine/tags/tags.jsonl from scratch (a few minutes). `verify` checks
that the stored patches are valid head inputs by recomputing a few songs from audio the reference
(Essentia) way. `report` prints per-playlist tag means against the library means. Details and the
output schema: bench/tags.md.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

CACHE = Path.home() / ".cache" / "audio-embed-engine"
OUT_DIR = CACHE / "tags"
MODELS = OUT_DIR / "models"
PATCHES = CACHE / "patches"
DB = CACHE / "embeddings.sqlite"
EFFNET = CACHE / "models" / "discogs-effnet-bsdynamic-1.onnx"  # downloaded by the engine
PREVIEW_MODEL = "discogs-effnet-1/mean-hop128/v1"
HEADS_URL = "https://essentia.upf.edu/models/classification-heads"
DIM = 1280

# Binary heads: output key -> (head dir, file stem, positive class as named in the head's metadata).
BINARY = {
    "happy": ("mood_happy", "mood_happy", "happy"),
    "sad": ("mood_sad", "mood_sad", "sad"),
    "relaxed": ("mood_relaxed", "mood_relaxed", "relaxed"),
    "aggressive": ("mood_aggressive", "mood_aggressive", "aggressive"),
    "party": ("mood_party", "mood_party", "party"),
    "acoustic": ("mood_acoustic", "mood_acoustic", "acoustic"),
    "electronic": ("mood_electronic", "mood_electronic", "electronic"),
    "danceable": ("danceability", "danceability", "danceable"),
    "approachable": ("approachability", "approachability_2c", "approachable"),
    "engaging": ("engagement", "engagement_2c", "engaging"),
    "voice": ("voice_instrumental", "voice_instrumental", "voice"),
    "female": ("gender", "gender", "female"),
    "bright": ("timbre", "timbre", "bright"),
    "tonal": ("tonal_atonal", "tonal_atonal", "tonal"),
}
MOOD = ("mtg_jamendo_moodtheme", "mtg_jamendo_moodtheme")
GENRE_TOP = 8


def fetch_head(d: str, stem: str) -> tuple[Path, list[str]]:
    """Download (once) an Essentia head's .onnx and .json; returns the model path and its class names."""
    MODELS.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("json", "onnx"):
        p = MODELS / f"{stem}-discogs-effnet-1.{ext}"
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(f"{HEADS_URL}/{d}/{p.name}", tmp)
            tmp.replace(p)
        out.append(p)
    return out[1], json.loads(out[0].read_text())["classes"]


@dataclass(frozen=True)
class Heads:
    """Every head, ready to run on a (patches, 1280) matrix of raw EffNet embeddings."""

    sessions: dict[str, tuple[ort.InferenceSession, int]]  # binary key -> (session, positive column)
    mood: ort.InferenceSession
    moods: list[str]
    genre_w: np.ndarray  # (400, 1280): EffNet's own style classifier, sigmoid(emb @ W.T + b)
    genre_b: np.ndarray
    styles: list[str]

    @staticmethod
    def load() -> Heads:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        def session(p: Path) -> ort.InferenceSession:
            return ort.InferenceSession(p, opts, ["CPUExecutionProvider"])

        sessions = {}
        for key, (d, stem, positive) in BINARY.items():
            path, classes = fetch_head(d, stem)
            sessions[key] = (session(path), classes.index(positive))
        mood_path, moods = fetch_head(*MOOD)
        # The 400-style output of EffNet is Gemm(embeddings) -> Sigmoid, so it can run on stored embeddings.
        # (The separate genre_discogs400 head is published only as a TensorFlow .pb.)
        g = onnx.load(EFFNET).graph
        gemm = [n for n in g.node if n.op_type == "Gemm"][-1]
        init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        styles = json.loads(EFFNET.with_suffix(".json").read_text())["classes"]
        return Heads(sessions, session(mood_path), moods, init[gemm.input[1]], init[gemm.input[2]], styles)

    def per_patch(self, emb: np.ndarray) -> dict[str, np.ndarray]:
        """Head outputs for every row: binary keys -> (n,), "moods" -> (n, 56), "styles" -> (n, 400)."""
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        out = {k: s.run(None, {s.get_inputs()[0].name: emb})[0][:, col] for k, (s, col) in self.sessions.items()}
        out["moods"] = self.mood.run(None, {self.mood.get_inputs()[0].name: emb})[0]
        out["styles"] = 1 / (1 + np.exp(-(emb @ self.genre_w.T + self.genre_b)))
        return out

    def record(self, mean: dict[str, np.ndarray]) -> dict:
        """Per-song means -> one tags.jsonl record (without key)."""
        top = np.argsort(mean["styles"])[::-1][:GENRE_TOP]
        rec: dict = {"genre_discogs400": [[self.styles[i], round(float(mean["styles"][i]), 3)] for i in top]}
        rec["mtg_jamendo_moodtheme"] = {m: round(float(v), 3) for m, v in zip(self.moods, mean["moods"])}
        rec |= {k: round(float(mean[k]), 3) for k in BINARY}
        return rec


def song_means(heads: Heads, emb: np.ndarray, sizes: list[int]) -> list[dict[str, np.ndarray]]:
    """Average per-patch head outputs over each song; `sizes` are consecutive row counts per song."""
    out = heads.per_patch(emb)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    sums = {k: np.add.reduceat(v, starts, axis=0) for k, v in out.items()}
    n = np.asarray(sizes, dtype=np.float32)
    return [{k: v[i] / n[i] for k, v in sums.items()} for i in range(len(sizes))]


def patch_index() -> list[dict]:
    return [json.loads(line) for line in open(PATCHES / "index.jsonl")]


def stored_patches() -> np.ndarray:
    return np.memmap(PATCHES / "patches.f32", dtype=np.float32, mode="r").reshape(-1, DIM)


def preview_vectors(skip: set[str]) -> dict[str, np.ndarray]:
    """L2-normalised 30 s preview vectors for songs that have no full-track patches."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT track_key, vector FROM embeddings WHERE model = ? AND status = 'ok'", (PREVIEW_MODEL,))
    return {k: np.frombuffer(v, np.float32) for k, v in rows if k not in skip}


def cmd_run() -> None:
    t0 = time.perf_counter()
    heads = Heads.load()
    index, data = patch_index(), stored_patches()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "tags.jsonl.part"
    norms = []
    with open(tmp, "w") as f:
        for b in range(0, len(index), 250):
            chunk = index[b : b + 250]
            lo, hi = chunk[0]["start"], chunk[-1]["start"] + chunk[-1]["n"]
            assert all(c["start"] == p["start"] + p["n"] for p, c in zip(chunk, chunk[1:])), "index not contiguous"
            emb = np.asarray(data[lo:hi])
            sizes = [c["n"] for c in chunk]
            for c, mean, s in zip(chunk, song_means(heads, emb, sizes), np.cumsum([0, *sizes])):
                norms.append(float(np.linalg.norm(emb[s : s + c["n"]].mean(0))))
                f.write(json.dumps({"key": c["key"], "n": c["n"], **heads.record(mean)}) + "\n")
            print(f"{b + len(chunk)}/{len(index)}", file=sys.stderr, end="\r")
        # Preview-only songs: the stored vector is an L2-normalised mean embedding, so it is rescaled to the
        # median norm of full-track mean embeddings and run through the heads as one "patch" (see tags.md).
        scale = float(np.median(norms))
        previews = preview_vectors({c["key"] for c in index})
        keys = sorted(previews)
        if keys:
            emb = np.stack([previews[k] for k in keys]) * scale
            for k, mean in zip(keys, song_means(heads, emb, [1] * len(keys))):
                f.write(json.dumps({"key": k, "n": 0, "preview": True, **heads.record(mean)}) + "\n")
    tmp.replace(OUT_DIR / "tags.jsonl")
    print(f"\n{len(index)} full tracks + {len(keys)} previews in {time.perf_counter() - t0:.0f} s "
          f"(preview scale {scale:.3f})", file=sys.stderr)


# ---------- verify: stored patches vs the reference Essentia path ----------


def decode(path: Path) -> np.ndarray:
    """Stereo at 48 kHz, channels averaged, then soxr HQ to 16 kHz: the engine's decode (bench/full_vs_preview.py).
    Not ffmpeg's `-ac 1`, which mixes (L+R)/sqrt(2): 3 dB louder, and EffNet embeddings move with level."""
    import soxr

    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"]
    pcm = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).mean(1)
    return soxr.resample(pcm, 48000, 16000, quality="HQ").astype(np.float32)


def essentia_mel(audio: np.ndarray) -> np.ndarray:
    import essentia.standard as es

    ex = es.TensorflowInputMusiCNN()
    return np.array([ex(fr) for fr in es.FrameGenerator(audio, frameSize=512, hopSize=256)], np.float32)


def to_patches(mel: np.ndarray, hop: int) -> np.ndarray:
    starts = range(0, len(mel) - 128 + 1, hop)
    return np.stack([mel[s : s + 128] for s in starts])


def cmd_verify(n: int, src: str) -> None:
    import essentia.standard as es

    heads = Heads.load()
    effnet = ort.InferenceSession(EFFNET, providers=["CPUExecutionProvider"])
    index, data = patch_index(), stored_patches()
    files = {d["key"]: d["file"] for line in open(CACHE / "features" / "sources.jsonl")
             for d in [json.loads(line)] if d.get("file")}
    rng = np.random.default_rng(0)
    picks = [index[i] for i in rng.choice(len(index), n * 2, replace=False) if index[i]["key"] in files][:n]
    norms = [float(np.linalg.norm(data[p["start"] : p["start"] + p["n"]].mean(0))) for p in index[::10]]
    scale = float(np.median(norms))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "list").write_text("\n".join(f"tracks/{files[p['key']]}" for p in picks))
        subprocess.run(["rsync", "-t", "--timeout=15", "--files-from", f"{tmp}/list", src, tmp], check=True)
        for p in picks:
            path = Path(tmp) / "tracks" / files[p["key"]]
            stored = np.asarray(data[p["start"] : p["start"] + p["n"]])
            # A: same patching as the store (hop 128), Essentia mel on the engine's decode -> row-wise match
            mine = effnet.run(["embeddings"], {"melspectrogram": to_patches(essentia_mel(decode(path)), 128)})[0]
            m = min(len(mine), len(stored))
            cos = np.sum(mine[:m] * stored[:m], 1) / np.linalg.norm(mine[:m], axis=1) / np.linalg.norm(stored[:m], axis=1)
            ratio = np.linalg.norm(stored[:m], axis=1) / np.linalg.norm(mine[:m], axis=1)
            # B: the Essentia reference: MonoLoader at 16 kHz, patch hop 62 (TensorflowPredictEffnetDiscogs default)
            ref_audio = es.MonoLoader(filename=str(path), sampleRate=16000, resampleQuality=4)()
            ref_emb = effnet.run(["embeddings"], {"melspectrogram": to_patches(essentia_mel(ref_audio), 62)})[0]
            ref = song_means(heads, ref_emb, [len(ref_emb)])[0]
            ours = song_means(heads, stored, [len(stored)])[0]
            # C: the preview shortcut: one rescaled L2-normalised mean vector instead of averaging over patches
            mean = stored.mean(0)
            short = song_means(heads, (mean / np.linalg.norm(mean) * scale)[None], [1])[0]
            rows.append((p["key"], len(stored), len(mine), float(np.median(cos)), float(cos.min()), float(np.median(ratio)), ref, ours, short))

    print(f"{'key':14} {'stored':>6} {'redo':>5} {'med cos':>7} {'min cos':>7} {'norm ratio':>10}  top style ref / stored")
    def top(d: dict[str, np.ndarray]) -> str:
        return heads.styles[int(np.argmax(d["styles"]))]

    for key, ns, nm, cmed, cmin, r, ref, ours, _ in rows:
        print(f"{key:14} {ns:6} {nm:5} {cmed:7.4f} {cmin:7.4f} {r:10.4f}  {top(ref)} / {top(ours)}")
    print(f"\nper head, over {len(rows)} songs: mean |ref - stored|, max, and the same for the preview shortcut")
    for k in [*BINARY, "moods", "styles"]:
        d = np.array([np.abs(ref[k] - ours[k]).mean() for *_, ref, ours, _s in rows])
        dm = np.array([np.abs(ref[k] - ours[k]).max() for *_, ref, ours, _s in rows])
        s = np.array([np.abs(ours[k] - short[k]).mean() for *_, ours, short in rows])
        sm = np.array([np.abs(ours[k] - short[k]).max() for *_, ours, short in rows])
        print(f"  {k:13} {d.mean():.3f}  max {dm.max():.3f}   shortcut {s.mean():.3f}  max {sm.max():.3f}")


# ---------- report: do the tags separate the owner's playlists? ----------


def load_tags() -> dict[str, dict]:
    return {d["key"]: d for line in open(OUT_DIR / "tags.jsonl") for d in [json.loads(line)]}


def scalars(t: dict) -> dict[str, float]:
    return {k: t[k] for k in BINARY}


def style_vec(ts: Iterable[dict]) -> dict[str, float]:
    """Mean probability per style, counting a style as 0 for songs where it is outside the stored top 8."""
    ts = list(ts)
    acc: dict[str, float] = {}
    for t in ts:
        for s, p in t["genre_discogs400"]:
            acc[s] = acc.get(s, 0) + p
    return {s: v / len(ts) for s, v in acc.items()}


def mood_vec(ts: Iterable[dict]) -> dict[str, float]:
    ts = list(ts)
    return {m: float(np.mean([t["mtg_jamendo_moodtheme"][m] for t in ts])) for m in ts[0]["mtg_jamendo_moodtheme"]}


def short(s: str) -> str:
    return s.split("---")[-1]


def cmd_report(cento: Path) -> None:
    tags = load_tags()
    lib = json.loads((cento / "library.json").read_text())
    meta = {d["key"]: d for line in open(cento / "tracks.jsonl") for d in [json.loads(line)]}
    engine_key = {sid: m["isrc"].upper() if m.get("isrc") else sid for sid, m in meta.items()}
    all_t = list(tags.values())
    lib_s, lib_m = style_vec(all_t), mood_vec(all_t)
    sc_keys = ["happy", "sad", "relaxed", "aggressive", "party", "danceable"]
    lib_sc = {k: float(np.mean([t[k] for t in all_t])) for k in BINARY}
    lib_sd = {k: float(np.std([t[k] for t in all_t])) for k in BINARY}
    print(f"library: {len(all_t)} songs ({sum('preview' in t for t in all_t)} preview-only)")
    print("library means: " + "  ".join(f"{k} {lib_sc[k]:.2f}±{lib_sd[k]:.2f}" for k in BINARY))
    print("library top styles: " + ", ".join(f"{short(s)} {v:.2f}" for s, v in sorted(lib_s.items(), key=lambda x: -x[1])[:8]))
    print("library top moods:  " + ", ".join(f"{m} {v:.2f}" for m, v in sorted(lib_m.items(), key=lambda x: -x[1])[:8]))

    by_size = sorted(lib["playlists"], key=lambda p: -len(p["trackIds"]))
    pls = by_size[:40] + [p for p in by_size[40:] if p["name"] in EXTRA_PLAYLISTS]
    spread = {k: [] for k in BINARY}
    for p in pls:
        ts = [tags[engine_key[s]] for s in p["trackIds"] if s in engine_key and engine_key[s] in tags]
        if len(ts) < 5:
            continue
        s, m = style_vec(ts), mood_vec(ts)
        print(f"\n## {p['name']}  ({len(ts)}/{len(p['trackIds'])} tagged)")
        # styles ranked by playlist mean; the lift over the library mean shows what is distinctive
        print("  styles: " + ", ".join(f"{short(k)} {v:.2f}/{lib_s.get(k, 0):.2f}"
                                     for k, v in sorted(s.items(), key=lambda x: -x[1])[:5]))
        print("  moods:  " + ", ".join(f"{k} {v:.2f}/{lib_m[k]:.2f}" for k, v in sorted(m.items(), key=lambda x: -x[1])[:5]))
        lift = sorted(m, key=lambda k: -(m[k] / max(lib_m[k], 1e-3)))[:3]
        print("  moods by lift: " + ", ".join(f"{k} x{m[k] / lib_m[k]:.1f}" for k in lift))
        means = {k: float(np.mean([t[k] for t in ts])) for k in BINARY}
        for k in BINARY:
            spread[k].append((means[k] - lib_sc[k]) / lib_sd[k])
        print("  " + "  ".join(f"{k} {means[k]:.2f}/{lib_sc[k]:.2f}" for k in sc_keys)
              + "   | " + "  ".join(f"{k} {means[k]:.2f}" for k in BINARY if k not in sc_keys))

    print("\nhow far playlist means move from the library mean, in library std units (max |z| over playlists, mean |z|):")
    for k, zs in sorted(spread.items(), key=lambda x: -np.max(np.abs(x[1]))):
        print(f"  {k:13} max {np.max(np.abs(zs)):.2f}  mean {np.mean(np.abs(zs)):.2f}")

    print("\nspot checks (top 3 styles | top 3 moods | happy sad aggressive danceable voice female):")
    by_title = {}
    for sid, m in meta.items():
        by_title.setdefault((m["artists"][0].lower(), m["title"].lower()), engine_key[sid])
    for artist, title in SPOT:
        hit = [(k, key) for k, key in by_title.items() if k[0] == artist.lower() and k[1].startswith(title.lower())]
        if not hit or hit[0][1] not in tags:
            print(f"  (not found: {artist} - {title})")
            continue
        (_, ttl), key = hit[0]
        t = tags[key]
        mo = sorted(t["mtg_jamendo_moodtheme"].items(), key=lambda x: -x[1])[:3]
        print(f"  {artist} - {ttl}: " + ", ".join(f"{short(s)} {p:.2f}" for s, p in t["genre_discogs400"][:3])
              + " | " + ", ".join(f"{k} {v:.2f}" for k, v in mo)
              + " | " + " ".join(f"{t[k]:.2f}" for k in ("happy", "sad", "aggressive", "danceable", "voice", "female")))


# Smaller playlists whose names imply a genre, mood or activity, reported after the 40 largest
EXTRA_PLAYLISTS = {"Future Bass", "Lo-fi", "J-Rock", "Gym", "Gaming", "Lock In", "Chillhop", "Rap",
                   "Regueton Antiguo", "NEO-PERREO", "Melodic Dubstep", "Indie", "Dance"}

# (primary artist, title prefix) picked from the owner's library across styles, for eyeballing
SPOT = [
    ("Bad Bunny", "DÁKITI"), ("Arcángel", "Infeliz"), ("Feid", "Ferxxo 100"), ("Linkin Park", "Numb"),
    ("Bring Me The Horizon", "Throne"), ("Pierce The Veil", "King For A Day"), ("YOASOBI", "アイドル"),
    ("MYTH & ROID", "STYX HELIX"), ("Sophie Ellis-Bextor", "Murder On The Dancefloor"),
    ("The Weeknd", "Blinding Lights"), ("Tame Impala", "The Less I Know The Better"), ("Billie Eilish", "bad guy"),
    ("Clairo", "Flaming Hot Cheetos"), ("Porter Robinson", "Look at the Sky"), ("ILLENIUM", "Good Things Fall Apart"),
    ("XXXTENTACION", "Looking for a Star"), ("Eden Project", "drowning"), ("STARSET", "ICARUS"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--src", default="audio-box:./")
    r = sub.add_parser("report")
    r.add_argument("cento_data", type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run()
    elif args.cmd == "verify":
        cmd_verify(args.n, args.src)
    else:
        cmd_report(args.cento_data)


if __name__ == "__main__":
    main()
