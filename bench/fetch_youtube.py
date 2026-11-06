"""Download full-track audio from YouTube Music for a track list, slowly, into a directory.

Usage: python bench/fetch_youtube.py <tracks.jsonl> <out_dir> [--sleep MIN MAX] [--limit N]

Each track is searched on YouTube Music (songs first, i.e. official "Topic" audio; see `find`)
and a result is accepted only if duration (±3 s), artist and title match, using the Deezer
matcher's rules. The best audio stream is saved untouched as `<cache_key>.<ext>` (Opus/WebM or AAC/M4A,
~3-5 MB per track). Progress goes to `<out_dir>/manifest.jsonl`, one line per finished track,
so the run resumes where it stopped. A bot-check answer from YouTube stops the run instead of
burning through the list.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import yt_dlp
from ytmusicapi import YTMusic

from audio_embed_engine.deezer import DURATION_TOLERANCE_S, _clean_title, _norm, _search_queries
from audio_embed_engine.types import Track, read_tracks

BOT_CHECK = ("confirm you're not a bot", "sign in to confirm")


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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    done = {json.loads(l)["key"] for l in manifest.open()} if manifest.exists() else set()
    todo = [t for t in dict.fromkeys(read_tracks(args.tracks)) if t.cache_key not in done][: args.limit]
    yt = YTMusic()
    print(f"{len(done)} already done, {len(todo)} to go", file=sys.stderr)

    start = time.perf_counter()
    with manifest.open("a") as log:
        for i, track in enumerate(todo, 1):
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
                row |= {"status": "error", "error": msg[:300]}
            row["seconds"] = round(time.perf_counter() - t0, 2)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"[{i}/{len(todo)}] {row['status']:9} {row['seconds']:5.1f}s  {track.artists[:1]} {track.title}",
                file=sys.stderr,
            )
            time.sleep(random.uniform(*args.sleep))
    took = time.perf_counter() - start
    print(f"{len(todo)} tracks attempted in {took / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
