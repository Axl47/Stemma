"""Download full-track audio from YouTube Music for a track list, slowly, into a directory.

Usage: python bench/fetch_youtube.py <tracks.jsonl> <out_dir> [--sleep MIN MAX] [--limit N]
                                    [--upload RSYNC_DEST [--upload-every N]]

Each track is searched on YouTube Music (songs first, i.e. official "Topic" audio; see `find`)
and a result is accepted only if duration (±3 s), artist and title match, using the Deezer
matcher's rules. The best audio stream is saved untouched as `<cache_key>.<ext>` (Opus/WebM or AAC/M4A,
~3-5 MB per track). Progress goes to `<out_dir>/manifest.jsonl`, one line per finished track,
so the run resumes where it stopped; `error` rows are retried on resume (`sign_in`, a per-video
gate, is not). A bot-check answer, or
MAX_CONSECUTIVE_ERRORS failures in a row (e.g. the network dropped), stops the run instead of
burning through the list.

With `--upload` (e.g. `audio-box:tracks/`), out_dir is only an outbox: every N tracks, and at the
end, finished audio files are rsynced to the destination and deleted locally once they've
arrived. The manifest stays local (it is what resume reads) and a copy goes along with the audio.
A failed upload is logged and retried at the next batch, so an unreachable box only grows the outbox.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yt_dlp
from ytmusicapi import YTMusic

from audio_embed_engine.deezer import DURATION_TOLERANCE_S, _clean_title, _norm, _search_queries
from audio_embed_engine.types import Track, read_tracks

BOT_CHECK = ("confirm you're not a bot", "sign in to confirm")
SIGN_IN = "please sign in"  # per-video gate (likely age restriction): permanent, not a network problem
MAX_CONSECUTIVE_ERRORS = 5
PARTIAL = (".part", ".ytdl")  # yt-dlp leftovers from an interrupted download


def upload(out_dir: Path, dest: str) -> None:
    """Move finished audio files to `dest` with rsync, then copy the manifest next to them."""
    files = sorted(
        p.name for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.jsonl" and p.suffix not in PARTIAL
    )
    try:
        if files:
            subprocess.run(
                ["rsync", "-t", "--remove-source-files", *files, dest], cwd=out_dir, check=True, timeout=1800
            )
        if (out_dir / "manifest.jsonl").exists():
            subprocess.run(["rsync", "-t", "manifest.jsonl", dest], cwd=out_dir, check=True, timeout=300)
        print(f"uploaded {len(files)} files to {dest}", file=sys.stderr)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"upload failed, will retry next batch: {e}", file=sys.stderr)


def match(track: Track, song: dict[str, Any]) -> bool:
    """Same rules as Deezer's search fallback, plus a title check: YouTube Music's song results
    often include the same artist's other songs with a similar length."""
    if abs((song.get("duration_seconds") or -999) - track.duration_ms / 1000) > DURATION_TOLERANCE_S:
        return False
    cand_artists = [_norm(a["name"]) for a in song.get("artists") or []]
    if not any(n and c and (n in c or c in n) for n in map(_norm, track.artists) for c in cand_artists):
        return False
    want, got = _norm(_clean_title(track.title)), _norm(song.get("title") or "")
    return bool(want and got and (want in got or got in want))


def find(yt: YTMusic, track: Track) -> tuple[str, dict[str, Any]] | None:
    """First match, trying in order: song search by artist + title, ISRC search (YouTube Music
    indexes some ISRCs), then video search (catches official-audio uploads on artist channels
    that the songs index misses). Returns (how it was found, search result)."""
    searches = [("song", lambda q=q: yt.search(q, filter="songs")) for q in _search_queries(track)]
    if track.isrc:
        searches.append(("isrc", lambda: yt.search(track.isrc or "")))
    searches += [("video", lambda q=q: yt.search(q, filter="videos")) for q in _search_queries(track)[:1]]
    for via, search in searches:
        for result in search():
            if result.get("videoId") and result.get("resultType") in ("song", "video") and match(track, result):
                return via, result
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--sleep", nargs=2, type=float, default=(3.0, 8.0), metavar=("MIN", "MAX"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--upload", metavar="RSYNC_DEST")
    ap.add_argument("--upload-every", type=int, default=25)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.open()] if manifest.exists() else []
    done = {r["key"] for r in rows if r["status"] in ("ok", "not_found", "sign_in")}
    unique = {t.cache_key: t for t in read_tracks(args.tracks)}  # releases sharing an ISRC are one recording
    todo = [t for key, t in unique.items() if key not in done][: args.limit]
    yt = YTMusic()
    print(f"{len(done)} already done, {len(todo)} to go", file=sys.stderr)

    start = time.perf_counter()
    errors_in_a_row = attempted = 0
    if args.upload:
        upload(args.out_dir, args.upload)  # anything a previous run left behind
    with manifest.open("a") as log:
        for i, track in enumerate(todo, 1):
            attempted = i
            t0 = time.perf_counter()
            row: dict[str, Any] = {"key": track.cache_key, "artists": track.artists, "title": track.title}
            try:
                found = find(yt, track)
                if found is None:
                    row["status"] = "not_found"
                else:
                    via, song = found
                    opts = {
                        "format": "bestaudio[acodec=opus]/bestaudio",
                        "outtmpl": str(args.out_dir / f"{track.cache_key}.%(ext)s"),
                        "quiet": True,
                        "no_warnings": True,
                        "noprogress": True,
                    }
                    with yt_dlp.YoutubeDL(opts) as ydl:  # pyright: ignore[reportArgumentType]
                        info = ydl.extract_info(f"https://music.youtube.com/watch?v={song['videoId']}")
                    path = Path(ydl.prepare_filename(info))
                    row |= {
                        "status": "ok",
                        "via": via,
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "video_id": song["videoId"],
                        "yt_title": song["title"],
                        "yt_artists": [a["name"] for a in song["artists"]],
                        "yt_duration": song["duration_seconds"],
                    }
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if any(s in msg.lower() for s in BOT_CHECK):
                    print(f"bot check after {i - 1} tracks, stopping: {msg}", file=sys.stderr)
                    break
                row |= {"status": "sign_in" if SIGN_IN in msg.lower() else "error", "error": msg[:300]}
            errors_in_a_row = errors_in_a_row + 1 if row["status"] == "error" else 0
            row["seconds"] = round(time.perf_counter() - t0, 2)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"[{i}/{len(todo)}] {row['status']:9} {row['seconds']:5.1f}s  {track.artists[:1]} {track.title}",
                file=sys.stderr,
            )
            if errors_in_a_row >= MAX_CONSECUTIVE_ERRORS:
                print(f"{errors_in_a_row} errors in a row (network down?), stopping", file=sys.stderr)
                break
            if args.upload and i % args.upload_every == 0:
                upload(args.out_dir, args.upload)
            time.sleep(random.uniform(*args.sleep))
    if args.upload:
        upload(args.out_dir, args.upload)
    took = time.perf_counter() - start
    print(f"{attempted} of {len(todo)} tracks attempted in {took / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
