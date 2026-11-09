"""Save per-patch (~2 s) Discogs-EffNet embeddings of the full tracks, for section-level search.

Usage: python bench/embed_patches.py <manifest.jsonl> <rsync_src> <out_dir> [--batch N] [--limit N]

Writes <out_dir>/patches.f32 (raw float32, 1280 per row, as the model outputs them) and
<out_dir>/index.jsonl ({"key", "start", "n"} per track, in file order). A track's rows are
patches.f32[start : start + n], in song order. Audio stays on the storage box: each batch is
rsynced to a temp dir and deleted, as in embed_full.py. Resumable: tracks in the index are skipped
and any bytes past the last indexed row (an interrupted write) are cut off.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from embed_full import fetch
from full_vs_preview import decode_file

from audio_embed_engine import DiscogsEffnet
from audio_embed_engine.audio import log_mel, patches

DIM, ROW_BYTES = 1280, 1280 * 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("src")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path, index_path = args.out_dir / "patches.f32", args.out_dir / "index.jsonl"
    done, rows = set[str](), 0
    if index_path.exists():
        for line in open(index_path):
            d = json.loads(line)
            done.add(d["key"])
            rows = d["start"] + d["n"]
    with open(data_path, "ab") as f:
        f.truncate(rows * ROW_BYTES)

    # latest row per key wins, as in embed_full.py
    files = {d["key"]: d.get("file") for line in open(args.manifest) for d in [json.loads(line)]}
    todo = sorted(k for k, f in files.items() if f and k not in done)[: args.limit]
    print(f"{len(todo)} tracks to do, {len(done)} done, {rows} rows", file=sys.stderr)

    model = DiscogsEffnet()
    effnet = model._sessions[0]
    start_t = time.perf_counter()
    with open(data_path, "ab") as data, open(index_path, "a") as index:
        for b in range(0, len(todo), args.batch):
            keys = todo[b : b + args.batch]
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "list").write_text("\n".join(files[k] for k in keys))
                fetch(f"{tmp}/list", args.src, f"{tmp}/a")
                for k in keys:
                    try:
                        mel = patches(log_mel(decode_file(Path(tmp) / "a" / files[k])), model.PATCH, model.HOP)
                        emb = np.asarray(effnet.run(["embeddings"], {"melspectrogram": mel})[0], dtype=np.float32)
                    except Exception as ex:  # noqa: BLE001
                        print(f"skip {k}: {type(ex).__name__}: {ex}", file=sys.stderr)
                        continue
                    assert emb.shape[1] == DIM
                    data.write(emb.tobytes())
                    data.flush()  # the index must never point at rows that aren't on disk
                    index.write(json.dumps({"key": k, "start": rows, "n": len(emb)}) + "\n")
                    index.flush()
                    rows += len(emb)
            n = b + len(keys)
            rate = (time.perf_counter() - start_t) / n
            print(f"{n}/{len(todo)}  {rate:.2f} s/track  ~{rate * (len(todo) - n) / 60:.0f} min left  {rows} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
