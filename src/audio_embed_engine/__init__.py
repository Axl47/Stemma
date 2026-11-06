"""Audio embedding engine: tracks -> Deezer 30 s previews -> Discogs-EffNet vectors, cached in SQLite.

Library use:
    from audio_embed_engine import Cache, DiscogsEffnet, embed_tracks, lookup, read_tracks
    tracks = read_tracks("tracks.jsonl"); cache = Cache("embeddings.sqlite"); model = DiscogsEffnet()
    embed_tracks(tracks, cache, model)
    results = lookup(tracks, cache, model.id)   # [Result(key, status, vector, tags)], input order
"""

from .cache import Cache
from .engine import EmbedStats, Result, embed_tracks, lookup
from .models import DiscogsEffnet, Embedding, Model
from .types import Status, Track, read_tracks

__all__ = [
    "Cache",
    "DiscogsEffnet",
    "EmbedStats",
    "Embedding",
    "Model",
    "Result",
    "Status",
    "Track",
    "embed_tracks",
    "lookup",
    "read_tracks",
]
