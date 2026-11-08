"""Embed the full YouTube tracks from bench/fetch_youtube.py into the cache under FULL_ID.

Usage: python bench/embed_full.py <manifest.jsonl> <rsync_src> [--db PATH] [--batch N]

The audio lives on the storage box (e.g. `audio-box:tracks/`), not the Mac: every batch is rsynced
into a temp dir, decoded, embedded with DiscogsEffnet (mean over the whole song) and deleted.
Rows go next to the preview rows under a separate model id, so `bench/library_eval.py --model`
can score either. Resumable: tracks already cached under FULL_ID are skipped.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from full_vs_preview import decode_file

from audio_embed_engine import Cache, DiscogsEffnet
from audio_embed_engine.cli import default_db

FULL_ID = DiscogsEffnet.id.replace("/v1", "/full-yt-v1")
FETCH_TIMEOUT_S = 180
ATTEMPTS = 5


def fetch(file_list: str, src: str, dest: str) -> None:
    """rsync a batch, on a fresh connection per attempt: a long-lived one to the box can stall
    without ever failing (seen: 2 h on one batch while a new connection took seconds)."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            subprocess.run(["rsync", "-t", "--files-from", file_list, src, dest], check=True, timeout=FETCH_TIMEOUT_S)
            return
        except (subprocess.SubprocessError, OSError) as e:
            print(f"fetch attempt {attempt}/{ATTEMPTS} failed: {e}", file=sys.stderr)
            if attempt == ATTEMPTS:
                raise
            time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("src")
    ap.add_argument("--db", default=str(default_db()))
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    # latest row per key wins: an earlier error row may have been retried successfully
    files = {d["key"]: d.get("file") for line in open(args.manifest) for d in [json.loads(line)]}
    cache = Cache(args.db)
    todo = sorted(k for k in cache.pending((k for k, f in files.items() if f), FULL_ID, retry_failed=True))
    print(f"{len(todo)} tracks to embed as {FULL_ID}", file=sys.stderr)
    model, start = DiscogsEffnet(), time.perf_counter()
    for b in range(0, len(todo), args.batch):
        keys = todo[b : b + args.batch]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "list").write_text("\n".join(files[k] for k in keys))
            fetch(f"{tmp}/list", args.src, f"{tmp}/a")
            for k in keys:
                try:
                    (e,) = model.embed([decode_file(Path(tmp) / "a" / files[k])])
                    cache.put_ok(k, FULL_ID, e.vector, e.tags, 0)
                except Exception as ex:  # noqa: BLE001
                    cache.put_failure(k, FULL_ID, "decode_error", f"{type(ex).__name__}: {ex}")
            cache.commit()
        done = b + len(keys)
        rate = (time.perf_counter() - start) / done
        print(f"{done}/{len(todo)}  {rate:.2f} s/track  ~{rate * (len(todo) - done) / 60:.0f} min left", file=sys.stderr)


if __name__ == "__main__":
    main()
