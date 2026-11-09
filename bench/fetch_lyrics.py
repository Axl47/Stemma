"""Fetch lyrics for a library from LRCLIB (free, no key), politely and resumably.

Usage: python bench/fetch_lyrics.py <tracks.jsonl> <out.jsonl> [--limit N]

Tracks are the engine's input lines; each unique isrc (or key) gets one output line:
{"key", "status": "ok" | "instrumental" | "not_found", "via": "get" | "search", "plain", "synced", "lrclib_id"}.
Lookup is the exact signature (title, first artist, duration ±2 s), then a search on a cleaned
title accepted only within ±3 s and with a matching artist. LRCLIB asks for sequential requests
with a delay and for Retry-After to be honoured; identified clients get ~90 requests/min.
Failures that may be temporary are not written, so a re-run retries them.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

API = "https://lrclib.net/api"
HEADERS = {"User-Agent": "audio-embed-engine/0.1 (personal music library tool)"}
GAP_S = 0.8  # between request starts: ~75/min, under the ~90/min limit
NOISE = re.compile(r"\s*[\(\[-].*?(feat\.|ft\.|with |remaster|version|edit|mix\b|live|mono|stereo).*$", re.IGNORECASE)

last_request = 0.0


def get(client: httpx.Client, path: str, params: dict[str, Any]) -> Any | None:
    """JSON body, or None on 404. Waits out 429/503/5xx/network errors, up to 8 tries."""
    global last_request
    for attempt in range(8):
        time.sleep(max(0.0, last_request + GAP_S - time.monotonic()))
        last_request = time.monotonic()
        try:
            r = client.get(f"{API}{path}", params=params)
        except httpx.TransportError:
            time.sleep(2**attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("retry-after", 2**attempt)) * (1 + random.random() * 0.3))
            continue
        raise RuntimeError(f"{path} {params} -> {r.status_code} {r.text[:200]}")
    raise RuntimeError(f"{path} {params}: gave up after retries")


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", "", s.casefold())


def lookup(client: httpx.Client, t: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    artists: list[str] = t.get("artists") or []
    if not t.get("title") or not artists:
        return None
    seconds = round(t["durationMs"] / 1000) if t.get("durationMs") else None
    params = {"track_name": t["title"], "artist_name": artists[0]}
    if seconds and 1 <= seconds <= 3600:
        hit = get(client, "/get", params | {"duration": seconds})
        if hit:
            return hit, "get"
    title = NOISE.sub("", t["title"]) or t["title"]
    wanted = {norm(a) for a in artists}
    for r in get(client, "/search", {"track_name": title, "artist_name": artists[0]}) or []:
        close = seconds is None or abs((r.get("duration") or 0) - seconds) <= 3
        if close and any(a and a in norm(r.get("artistName") or "") for a in wanted):
            return r, "search"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    done = {json.loads(line)["key"] for line in open(args.out)} if args.out.exists() else set()
    unique: dict[str, dict[str, Any]] = {}
    for line in open(args.tracks):
        t = json.loads(line)
        unique.setdefault(t.get("isrc") or t["key"], t)
    todo = [(k, t) for k, t in unique.items() if k not in done][: args.limit]
    print(f"{len(todo)} to fetch, {len(done)} done", file=sys.stderr)

    counts: dict[str, int] = {}
    with httpx.Client(headers=HEADERS, timeout=30) as client, open(args.out, "a") as out:
        for n, (key, t) in enumerate(todo, 1):
            found = lookup(client, t)
            if found is None:
                row: dict[str, Any] = {"key": key, "status": "not_found"}
            else:
                r, via = found
                row = {
                    "key": key,
                    "status": "instrumental" if r.get("instrumental") else "ok",
                    "via": via,
                    "plain": r.get("plainLyrics"),
                    "synced": r.get("syncedLyrics"),
                    "lrclib_id": r.get("id"),
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
