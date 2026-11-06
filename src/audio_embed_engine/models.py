"""Embedding models. The engine only needs the `Model` protocol; `DiscogsEffnet` is the default.

A model's `id` namespaces cache rows, so it must change whenever its vectors would change
(weights, pooling, preprocessing). Bump the trailing version when editing `embed`.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import onnxruntime as ort

from .audio import SAMPLE_RATE, Audio, log_mel, patches
from .cache import Tags

MODELS_URL = "https://essentia.upf.edu/models"


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray  # float32, L2-normalised
    tags: Tags | None = None


class Model(Protocol):
    id: str
    sample_rate: int

    def embed(self, audios: list[Audio]) -> list[Embedding]: ...


@dataclass(frozen=True)
class _Weights:
    path: str  # relative to MODELS_URL
    sha256: str


EFFNET = _Weights(
    "feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1",
    "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
)
MOOD_HEAD = _Weights(
    "classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1",
    "7d6270acaa5f4bba4b115a0d6849aca05ed6bd153dcb6d9da4f6ab9f99ef10ff",
)


def default_model_dir() -> Path:
    return Path(os.environ.get("AUDIO_EMBED_MODELS", Path.home() / ".cache" / "audio-embed-engine" / "models"))


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, timeout=30, follow_redirects=True) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        f.writelines(r.iter_bytes(1 << 16))
    tmp.replace(dest)


def _fetch(w: _Weights, model_dir: Path) -> tuple[Path, list[str]]:
    """Download (once) and verify an Essentia ONNX model; returns its path and class names."""
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx, meta = model_dir / f"{Path(w.path).name}.onnx", model_dir / f"{Path(w.path).name}.json"
    if not meta.exists():
        _download(f"{MODELS_URL}/{w.path}.json", meta)
    if not onnx.exists():
        _download(f"{MODELS_URL}/{w.path}.onnx", onnx)
        if hashlib.sha256(onnx.read_bytes()).hexdigest() != w.sha256:
            onnx.unlink()
            raise RuntimeError(f"checksum mismatch for {w.path}.onnx")
    return onnx, json.loads(meta.read_text())["classes"]


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / max(float(np.linalg.norm(x)), 1e-12)).astype(np.float32)


def _top(scores: np.ndarray, names: list[str], k: int) -> dict[str, float]:
    return {names[i]: round(float(scores[i]), 4) for i in np.argsort(scores)[::-1][:k]}


class DiscogsEffnet:
    """Essentia Discogs-EffNet (ONNX). Vector = mean of 1280-d per-patch embeddings over
    non-overlapping 128-frame (~2 s) patches. Tags: top Discogs-400 styles + MTG-Jamendo moods."""

    id = "discogs-effnet-1/mean-hop128/v1"
    sample_rate = SAMPLE_RATE
    PATCH, HOP = 128, 128  # overlap (hop 62) doubled cost with identical benchmark quality

    def __init__(self, model_dir: Path | None = None, top_k: int = 5) -> None:
        """Cheap: weights are downloaded/loaded on the first `embed` call."""
        self.model_dir = model_dir or default_model_dir()
        self.top_k = top_k

    @cached_property
    def _sessions(self) -> tuple[ort.InferenceSession, list[str], ort.InferenceSession, list[str]]:
        effnet, styles = _fetch(EFFNET, self.model_dir)
        mood, moods = _fetch(MOOD_HEAD, self.model_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        cpu = ["CPUExecutionProvider"]  # CoreML EP was slower in the benchmark, see bench/REPORT.md
        return ort.InferenceSession(effnet, opts, cpu), styles, ort.InferenceSession(mood, opts, cpu), moods

    def embed(self, audios: list[Audio]) -> list[Embedding]:
        effnet, styles, mood, moods = self._sessions
        per_track = [patches(log_mel(a), self.PATCH, self.HOP) for a in audios]
        style_act, emb = map(
            np.asarray, effnet.run(["activations", "embeddings"], {"melspectrogram": np.concatenate(per_track)})
        )
        mood_act = np.asarray(mood.run(["activations"], {mood.get_inputs()[0].name: emb})[0])
        out, start = [], 0
        for p in per_track:
            s = slice(start, start + len(p))
            start += len(p)
            tags = {
                "styles": _top(style_act[s].mean(0), styles, self.top_k),
                "moods": _top(mood_act[s].mean(0), moods, self.top_k),
            }
            out.append(Embedding(_l2(emb[s].mean(0)), tags))
        return out
