"""Compare candidate embedding models on speed and on "does it put my playlists together".

Usage:
  python bench/benchmark.py <models_dir> <sample.npz> <sample-tracks.jsonl> <sample-labels.jsonl> \
      [--fixture fixture.npz bench/tracks.jsonl] [--only name,name]

The .npz files hold raw preview MP3 bytes from bench/fetch_audio.py. Quality is judged on
pairs/neighbours that share NO artist, because the sorter's artist signal already covers
the rest; see bench/metrics.py for the exact definitions.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from metrics import pair_auc, playlist_ranking, style_knn

from audio_embed_engine.audio import decode, log_mel, patches

type Embedder = Callable[[list[np.ndarray]], np.ndarray]  # audios at `rate` -> (n, d)


@dataclass
class Candidate:
    name: str
    rate: int
    load: Callable[[], Embedder]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def essentia_onnx(
    path: Path,
    patch: int,
    hop: int,
    output: str,
    fixed_batch: int | None = None,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> Callable[[], Embedder]:
    """Mean-pooled per-patch output of an Essentia ONNX model fed with our numpy mel front end."""

    def load() -> Embedder:
        sess = ort.InferenceSession(str(path), providers=list(providers))
        inp = sess.get_inputs()[0].name

        def run(batch: np.ndarray) -> np.ndarray:
            if fixed_batch is None:
                return sess.run([output], {inp: batch})[0]
            outs = []
            for i in range(0, len(batch), fixed_batch):
                chunk = batch[i : i + fixed_batch]
                padded = np.zeros((fixed_batch, *chunk.shape[1:]), np.float32)
                padded[: len(chunk)] = chunk
                outs.append(sess.run([output], {inp: padded})[0][: len(chunk)])
            return np.concatenate(outs)

        def embed(audios: list[np.ndarray]) -> np.ndarray:
            ps = [patches(log_mel(a), patch, hop) for a in audios]
            out = run(np.ascontiguousarray(np.concatenate(ps)))
            bounds = np.cumsum([0] + [len(p) for p in ps])
            return np.stack([out[s:e].mean(0) for s, e in zip(bounds[:-1], bounds[1:])])

        return embed

    return load


def clap(checkpoint: str) -> Callable[[], Embedder]:
    # NB: laion/larger_clap_music gives collapsed embeddings (mean pairwise cos 0.94, zero-shot
    # labels everything "metal") under transformers 5.22; clap-htsat-unfused behaves normally.
    return lambda: _clap(checkpoint)


def _clap(checkpoint: str) -> Embedder:
    import torch
    from transformers import ClapModel, ClapProcessor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ClapModel.from_pretrained(checkpoint).to(dev).eval()
    proc = ClapProcessor.from_pretrained(checkpoint)

    def embed(audios: list[np.ndarray]) -> np.ndarray:
        out = []
        for a in audios:  # 10 s chunks (the model's window), averaged
            chunks = [a[i : i + 480_000] for i in range(0, max(len(a) - 240_000, 1), 480_000)]
            feats = proc(audio=chunks, sampling_rate=48_000, return_tensors="pt").to(dev)
            with torch.no_grad():
                e = model.get_audio_features(**feats)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            out.append(e.float().mean(0).cpu().numpy())
        return np.stack(out)

    return embed


def mert() -> Embedder:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(dev).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # The remote MERT code ignores output_hidden_states under transformers 5, so hook each layer.
    layer_out: list = []
    for layer in model.encoder.layers:
        layer.register_forward_hook(lambda _m, _i, o: layer_out.append(o[0] if isinstance(o, tuple) else o))

    def embed(audios: list[np.ndarray]) -> np.ndarray:
        out = []
        for a in audios:
            x = fe(a, sampling_rate=24_000, return_tensors="pt").to(dev)
            layer_out.clear()
            with torch.no_grad():
                model(**x)
            out.append(
                torch.stack(layer_out).mean(dim=(0, 2)).squeeze(0).float().cpu().numpy()
            )  # mean over layers + time
        return np.stack(out)

    return embed


def candidates(models: Path) -> list[Candidate]:
    eff = models / "discogs-effnet-bsdynamic-1.onnx"
    cands = [
        Candidate("effnet", 16_000, essentia_onnx(eff, 128, 62, "embeddings")),
        Candidate("effnet-hop128", 16_000, essentia_onnx(eff, 128, 128, "embeddings")),
        Candidate("effnet-styles400", 16_000, essentia_onnx(eff, 128, 62, "activations")),
        Candidate(
            "effnet-coreml",
            16_000,
            essentia_onnx(eff, 128, 62, "embeddings", providers=("CoreMLExecutionProvider", "CPUExecutionProvider")),
        ),
        Candidate("musicnn", 16_000, essentia_onnx(models / "msd-musicnn-1.onnx", 187, 93, "embeddings")),
        Candidate("clap-music", 48_000, clap("laion/larger_clap_music")),
        Candidate("clap-htsat-unfused", 48_000, clap("laion/clap-htsat-unfused")),
        Candidate("mert", 24_000, mert),
    ]
    for kind in ("artist", "track", "release", "label", "multi"):
        path = models / f"discogs_{kind}_embeddings-effnet-bs64-1.onnx"
        cands.append(Candidate(f"discogs-{kind}", 16_000, essentia_onnx(path, 128, 62, "embeddings", fixed_batch=64)))
    return cands


def embed_all(c: Candidate, mp3s: dict[str, bytes], batch: int = 16) -> tuple[list[str], np.ndarray, float, float]:
    embed = c.load()
    keys = list(mp3s)
    embed([decode(mp3s[keys[0]], c.rate)])  # warm-up (graph compile, MPS init)
    t_dec = t_inf = 0.0
    out = []
    for i in range(0, len(keys), batch):
        t = time.perf_counter()
        audios = [decode(mp3s[k], c.rate) for k in keys[i : i + batch]]
        t_dec += time.perf_counter() - t
        t = time.perf_counter()
        out.append(embed(audios))
        t_inf += time.perf_counter() - t
    return keys, l2(np.concatenate(out)), t_dec, t_inf


def load_npz(path: str) -> dict[str, bytes]:
    with np.load(path) as z:
        return {k: z[k].tobytes() for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path)
    ap.add_argument("sample")
    ap.add_argument("tracks")
    ap.add_argument("labels")
    ap.add_argument("--fixture", nargs=2)
    ap.add_argument("--only")
    ap.add_argument("--out", default="bench/results.json")
    args = ap.parse_args()

    tracks = {(d := json.loads(l)).get("isrc") or d["key"]: d for l in open(args.tracks)}
    playlists = {d["key"]: d["playlists"] for l in open(args.labels) for d in [json.loads(l)]}
    sample = load_npz(args.sample)
    artists = {k: set(tracks[k]["artists"]) for k in sample}
    labels = {k: set(playlists[tracks[k]["key"]]) for k in sample}
    fixture = load_npz(args.fixture[0]) if args.fixture else {}
    fx_tracks = {(d := json.loads(l))["isrc"]: d for l in open(args.fixture[1])} if args.fixture else {}

    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    only = set(args.only.split(",")) if args.only else None
    rng = np.random.default_rng(0)
    for c in [Candidate("random", 16_000, lambda: lambda a: rng.normal(size=(len(a), 256)))] + candidates(args.models):
        if only and c.name not in only:
            continue
        keys, emb, t_dec, t_inf = embed_all(c, sample)
        r = {
            "dim": int(emb.shape[1]),
            "decode_ms_per_track": 1000 * t_dec / len(keys),
            "infer_ms_per_track": 1000 * t_inf / len(keys),
            "pair_auc_no_shared_artist": pair_auc(keys, emb, labels, artists, exclude_shared_artist=True),
            "pair_auc_all": pair_auc(keys, emb, labels, artists, exclude_shared_artist=False),
            **playlist_ranking(keys, emb, labels, artists),
        }
        if fixture:
            fkeys, femb, _, _ = embed_all(c, fixture)
            r |= style_knn(
                fkeys, femb, {k: fx_tracks[k]["style"] for k in fkeys}, {k: fx_tracks[k]["artistLabel"] for k in fkeys}
            )
        results[c.name] = r
        print(c.name, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
