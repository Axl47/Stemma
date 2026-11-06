"""Full-library check: can embeddings place a track by an artist the user has never sorted?

Usage: python bench/library_eval.py <library.json> <tracks.jsonl> [--db PATH]

Every track that sits in at least one playlist becomes a leave-one-out query. Playlists are
scored only from members that share NO artist (by Spotify artist id) with the query, which
simulates a song by a never-sorted artist. Reports recall@3 (share of the track's playlists
in the top 3), hit@3, and a popularity baseline (always suggest the 3 biggest playlists).
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from audio_embed_engine import Cache, DiscogsEffnet, lookup, read_tracks
from audio_embed_engine.cli import default_db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("tracks")
    ap.add_argument("--db", default=str(default_db()))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    lib = json.load(open(args.library))
    results = {r.key: r.vector for r in lookup(read_tracks(args.tracks), Cache(args.db), DiscogsEffnet.id)}
    keys = [k for k, v in results.items() if v is not None]
    index = {k: i for i, k in enumerate(keys)}
    emb = np.stack([results[k] for k in keys])
    artists = [frozenset(a["id"] for a in lib["tracks"][k]["artists"]) for k in keys]

    playlists: dict[str, list[int]] = {}
    for p in lib["playlists"]:
        members = sorted({index[t] for t in p["trackIds"] if t in index})
        if members:
            playlists[p["name"]] = members
    names = list(playlists)
    member_of: dict[int, set[int]] = defaultdict(set)
    for pi, name in enumerate(names):
        for i in playlists[name]:
            member_of[i].add(pi)
    by_size = sorted(range(len(names)), key=lambda pi: -len(playlists[names[pi]]))

    by_artist: dict[str, set[int]] = defaultdict(set)
    for i, a in enumerate(artists):
        for aid in a:
            by_artist[aid].add(i)

    k = args.k
    stats: dict[str, list[float]] = defaultdict(list)
    for q, truth_all in member_of.items():
        excluded = set().union(*(by_artist[a] for a in artists[q]))  # includes q itself
        sims = emb @ emb[q]
        scores = {"centroid": np.full(len(names), -np.inf), "top5": np.full(len(names), -np.inf)}
        for pi, name in enumerate(names):
            ms = [i for i in playlists[name] if i not in excluded]
            if not ms:
                continue
            c = emb[ms].mean(0)
            scores["centroid"][pi] = float(emb[q] @ c / np.linalg.norm(c))
            scores["top5"][pi] = float(np.sort(sims[ms])[-5:].mean())
        truth = {pi for pi in truth_all if np.isfinite(scores["centroid"][pi])}
        if not truth:
            continue  # every playlist it's in is made only of its own artists (e.g. artist playlists)
        candidates = int(np.isfinite(scores["centroid"]).sum())
        for name, s in scores.items():
            top = set(np.argsort(s)[::-1][:k].tolist())
            stats[f"{name}_recall@{k}"].append(len(top & truth) / len(truth))
            stats[f"{name}_hit@{k}"].append(float(bool(top & truth)))
        popular = {pi for pi in by_size if np.isfinite(scores["centroid"][pi])}
        top_pop = set([pi for pi in by_size if pi in popular][:k])
        stats[f"popular_recall@{k}"].append(len(top_pop & truth) / len(truth))
        stats[f"random_recall@{k}"].append(min(k / candidates, 1.0))

    print(
        json.dumps(
            {"queries": len(stats[f"centroid_recall@{k}"]), "embedded_tracks": len(keys), "playlists": len(names)}
            | {n: round(float(np.mean(v)), 3) for n, v in stats.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
