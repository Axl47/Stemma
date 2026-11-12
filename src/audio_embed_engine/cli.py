"""`audio-embed` command line: `embed` fills the cache, `export` reads vectors back out, `classify`
trains per-playlist classifiers from those vectors, and `languages` labels lyrics for `classify`.

Progress goes to stderr; machine-readable output (embed summary, export rows) to stdout or --out.
"""

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

import numpy as np

from .cache import Cache
from .classifier import CONFIDENCE, LYRICS_WEIGHT, MIN_TRUST, train, with_block, with_language
from .engine import EmbedStats, Result, embed_tracks, lookup
from .language import detect_language
from .models import DiscogsEffnet
from .types import read_tracks


def default_db() -> Path:
    return Path(os.environ.get("AUDIO_EMBED_DB", Path.home() / ".cache" / "audio-embed-engine" / "embeddings.sqlite"))


def _progress(total: int):
    def report(s: EmbedStats) -> None:
        done = sum(s.statuses.values())
        print(f"\r{done}/{total} processed ({s.statuses['ok']} ok)", end="", file=sys.stderr, flush=True)

    return report


def cmd_embed(args: argparse.Namespace) -> None:
    tracks = read_tracks(args.tracks)
    model = DiscogsEffnet()
    cache = Cache(args.db)
    todo = len(cache.pending({t.cache_key for t in tracks}, model.id, args.retry_failed))
    stats = embed_tracks(
        tracks,
        cache,
        model,
        workers=args.workers,
        batch_size=args.batch,
        retry_failed=args.retry_failed,
        on_progress=None if args.quiet else _progress(todo),
    )
    if not args.quiet and todo:
        print(file=sys.stderr)
    print(
        json.dumps(
            {
                "model": model.id,
                "unique_tracks": stats.total,
                "cached": stats.cached,
                "processed": dict(stats.statuses),
                "api_calls": stats.api_calls,
                "inference_s": round(stats.inference_s, 2),
                "wall_s": round(stats.wall_s, 2),
            }
        )
    )


def _write_npy(results: list[Result], prefix: str) -> None:
    """PREFIX.npy: (n_ok, dim) float32. PREFIX.index.jsonl: {key, status, row} per input line (row null if no vector)."""
    vectors: list[np.ndarray] = []
    with open(f"{prefix}.index.jsonl", "w") as f:
        for r in results:
            row_index = None
            if r.vector is not None:
                row_index = len(vectors)
                vectors.append(r.vector)
            f.write(json.dumps({"key": r.key, "status": r.status, "row": row_index}) + "\n")
    np.save(f"{prefix}.npy", np.stack(vectors) if vectors else np.zeros((0, 0), np.float32))


def _write_jsonl(results: list[Result], out: TextIO, tags: bool) -> None:
    for r in results:
        row: dict[str, object] = {"key": r.key, "status": r.status}
        if r.vector is not None:
            row["vector"] = [round(float(x), 6) for x in r.vector]
        if tags and r.tags:
            row["tags"] = r.tags
        out.write(json.dumps(row) + "\n")


def prefer(preferred: list[Result], fallback: list[Result]) -> list[Result]:
    """Per line of the same track list: the preferred result if it has a vector, else the fallback."""
    return [p if p.vector is not None else f for p, f in zip(preferred, fallback, strict=True)]


def cmd_export(args: argparse.Namespace) -> None:
    tracks, cache = read_tracks(args.tracks), Cache(args.db)
    results = lookup(tracks, cache, DiscogsEffnet.id)
    if args.prefer_model:
        results = prefer(lookup(tracks, cache, args.prefer_model), results)
    if args.format == "npy":
        if not args.out:
            sys.exit("--format npy needs --out PREFIX")
        _write_npy(results, args.out)
        return
    with open(args.out, "w") if args.out else nullcontext(sys.stdout) as out:
        _write_jsonl(results, out, args.tags)


def _read_block(path: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-song vectors from JSONL `{"key", "vector"}`, as rows aligned to `keys`, scaled to unit length.
    Songs missing from the file get a zero row and a False flag."""
    wanted = set(keys)
    found: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            v = np.asarray(row["vector"], np.float32)
            if row["key"] in wanted and (norm := np.linalg.norm(v)) > 0:
                found[row["key"]] = v / norm
    if not found:
        sys.exit(f"{path}: no vectors for any song in the labels file")
    block = np.zeros((len(keys), len(next(iter(found.values())))), np.float32)
    present = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        if k in found:
            block[i], present[i] = found[k], True
    return block, present


def cmd_classify(args: argparse.Namespace) -> None:
    vectors: dict[str, list[float]] = {}
    with open(args.vectors) as f:
        for line in f:
            row = json.loads(line)
            if row.get("vector"):
                vectors[row["key"]] = row["vector"]
    songs: dict[str, dict] = {}
    with open(args.labels) as f:
        for line in f:
            row = json.loads(line)
            if row["key"] in vectors:
                songs.setdefault(row["key"], row)
    keys = list(songs)
    counts = Counter(label for s in songs.values() for label in s["labels"])
    playlists = sorted(p for p, n in counts.items() if n >= args.min_size)
    column = {p: i for i, p in enumerate(playlists)}

    x = np.array([vectors[k] for k in keys], np.float32)
    y = np.zeros((len(keys), len(playlists)), bool)
    for i, k in enumerate(keys):
        for label in songs[k]["labels"]:
            if label in column:
                y[i, column[label]] = True
    # a song with no listed artist still needs a fold of its own
    artists = [frozenset(songs[k]["artists"]) or frozenset({k}) for k in keys]
    primary = [(songs[k]["artists"] or [k])[0] for k in keys]
    x, languages = with_language(x, [songs[k].get("language") for k in keys], y.any(1))
    if args.lyrics:
        block, present = _read_block(args.lyrics, keys)
        x = with_block(x, block, present, y.any(1), weight=args.lyrics_weight)
    scores, picks, fits = train(x, y, artists, primary, confidence=args.confidence, min_trust=args.min_trust)

    output = {
        "confidence": args.confidence,
        "languages": languages,
        "keys": keys,
        "playlists": [
            {"id": p, "auc": None if np.isnan(f.auc) else round(f.auc, 3), "cutoff": f.cutoff}
            for p, f in zip(playlists, fits, strict=True)
        ],
        "scores": np.round(scores, 3).tolist(),
        "picks": picks,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    summary = {
        "songs": len(keys),
        "labelled": int(y.any(1).sum()),
        "playlists": len(playlists),
        "languages": languages,
        "can_auto_pick": sum(f.cutoff is not None for f in fits),
        "auto_picks": sum(p is not None for p in picks),
    }
    print(json.dumps(summary))


def cmd_languages(args: argparse.Namespace) -> None:
    found = 0
    with open(args.lyrics) as f, open(args.out, "w") as out:
        for line in f:
            row = json.loads(line)
            if row["status"] == "ok" and (language := detect_language(row.get("plain") or "")):
                out.write(json.dumps({"key": row["key"], "language": language}) + "\n")
                found += 1
    print(json.dumps({"songs_with_language": found}))


def main() -> None:
    ap = argparse.ArgumentParser(prog="audio-embed", description=__doc__)
    ap.add_argument("--db", type=Path, default=default_db(), help="SQLite cache (env AUDIO_EMBED_DB)")
    sub = ap.add_subparsers(required=True)

    e = sub.add_parser("embed", help="embed every track in TRACKS not already cached")
    e.add_argument("tracks", help="JSONL: {isrc, title, artists, durationMs, key?} per line")
    e.add_argument("--workers", type=int, default=16, help="concurrent Deezer fetches")
    e.add_argument("--batch", type=int, default=32, help="tracks per inference batch")
    e.add_argument("--retry-failed", action="store_true", help="ignore retry-after on cached failures")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_embed)

    x = sub.add_parser("export", help="write cached results for every line of TRACKS, in order")
    x.add_argument("tracks")
    x.add_argument("--format", choices=["jsonl", "npy"], default="jsonl")
    x.add_argument("--out", help="file (jsonl; default stdout) or PREFIX for PREFIX.npy + PREFIX.index.jsonl")
    x.add_argument("--tags", action="store_true", help="include style/mood tags (jsonl only)")
    x.add_argument("--prefer-model", help="use this model id's vector where cached, the default model's elsewhere")
    x.set_defaults(func=cmd_export)

    c = sub.add_parser("classify", help="train a sound classifier per playlist and score every song")
    c.add_argument(
        "labels",
        help='JSONL: {"key", "artists": [ids], "labels": [playlist ids], "language"?: code} per line; labels may be empty',
    )
    c.add_argument("--vectors", required=True, help="JSONL from `export`")
    c.add_argument("--out", required=True)
    c.add_argument("--confidence", type=float, default=CONFIDENCE, help="chance of belonging needed to auto-pick")
    c.add_argument("--min-trust", type=float, default=MIN_TRUST, help="lowest held-out playlist AUC that may auto-pick")
    c.add_argument(
        "--lyrics", help='JSONL: {"key", "vector"} per line, one vector per song (e.g. mean lyric passage embedding)'
    )
    c.add_argument(
        "--lyrics-weight",
        type=float,
        default=LYRICS_WEIGHT,
        help="how much the lyric vectors count next to sound (1 = equal)",
    )
    c.add_argument("--min-size", type=int, default=12, help="ignore playlists with fewer embedded songs")
    c.set_defaults(func=cmd_classify)

    g = sub.add_parser("languages", help="guess each song's language from its fetched lyrics")
    g.add_argument("lyrics", help='JSONL: {"key", "status", "plain"} per line, as from bench/fetch_lyrics.py')
    g.add_argument("--out", required=True, help='JSONL: {"key", "language"} for each song with a confident guess')
    g.set_defaults(func=cmd_languages)

    args = ap.parse_args()
    args.func(args)
