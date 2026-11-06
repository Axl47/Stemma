"""Fetch + decode previews for a track list into a temporary .npz (benchmark only; delete after).

Usage: python bench/fetch_audio.py <tracks.jsonl> <out.npz>
Prints Deezer hit rate and timing. Stores the raw MP3 bytes so each model can decode at its own rate.
"""

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from audio_embed_engine.audio import DecodeError, decode
from audio_embed_engine.deezer import Deezer, Preview
from audio_embed_engine.types import Track, read_tracks


def main(tracks_path: str, out_path: str) -> None:
    tracks = read_tracks(tracks_path)
    dz = Deezer()
    statuses: Counter[str] = Counter()
    audio: dict[str, np.ndarray] = {}

    def work(t: Track) -> tuple[Track, str, np.ndarray | None]:
        try:
            res = dz.resolve(t)
            if not isinstance(res, Preview):
                return t, res, None
            data = dz.download(res)
            decode(data)  # surface decode errors here, same as the engine would
            return t, "ok", np.frombuffer(data, dtype=np.uint8)
        except DecodeError:
            return t, "decode_error", None
        except Exception as e:  # noqa: BLE001
            print("error", t.title, e, file=sys.stderr)
            return t, "error", None

    start = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        for t, status, a in ex.map(work, tracks):
            statuses[status] += 1
            if a is not None:
                audio[t.cache_key] = a
            elif status != "ok":
                print(status, t.cache_key, t.artists, t.title, file=sys.stderr)
    took = time.perf_counter() - start
    np.savez(out_path, **audio)
    print(f"{len(tracks)} tracks in {took:.1f}s ({len(tracks) / took * 60:.0f}/min), api calls {dz.api_calls}")
    print(dict(statuses), f"hit rate {statuses['ok'] / len(tracks):.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
