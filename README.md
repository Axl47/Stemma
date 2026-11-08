<img src="assets/icon.svg" width="64" height="64" alt="">

# audio-embed-engine

Turns tracks into "how it sounds" vectors. A track's ISRC is looked up on Deezer, its 30 s
preview is downloaded and decoded in memory, and a local music model
([Essentia Discogs-EffNet](https://essentia.upf.edu/models.html#discogs-effnet), ONNX) turns it into
a 1280-d L2-normalised vector plus a few style and mood tags. Results are cached in SQLite,
so each track is processed once per model.

Nothing here is Spotify-specific. Callers pass track metadata in and read vectors out.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 is fetched by uv if missing.

```sh
cd /Users/axel/.t3/projects/audio-embed-engine
uv sync
```

Model weights (~21 MB) download to `~/.cache/audio-embed-engine/models` on first use and are
checked against a pinned SHA-256. No system packages are needed: decoding uses the libsndfile
bundled in the `soundfile` wheel, and the mel front end is numpy.

## Calling it from another project (subprocess)

Run from any cwd. Relative paths resolve against *your* cwd, not this repo:

```sh
uv run --project /Users/axel/.t3/projects/audio-embed-engine audio-embed embed data/tracks.jsonl
uv run --project /Users/axel/.t3/projects/audio-embed-engine audio-embed export data/tracks.jsonl --out data/embeddings.jsonl
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `AUDIO_EMBED_DB` | `~/.cache/audio-embed-engine/embeddings.sqlite` | Cache file (also `--db PATH`, placed before the subcommand) |
| `AUDIO_EMBED_MODELS` | `~/.cache/audio-embed-engine/models` | Where model weights live |

Exit code is 0 when the run completes, even if some tracks failed. Failures are data, reported
per track. A non-zero exit means the run itself broke (bad input file, unreadable cache).

### Input: JSONL, one track per line

```json
{"isrc": "GBAKW0100229", "title": "Murder On The Dancefloor", "artists": ["Sophie Ellis-Bextor"], "durationMs": 230013, "key": "4tKGFmENO69tZR9ahgZu48"}
```

- `isrc` (string or null) is the cache key. Several releases of one recording share an ISRC
  and are embedded once.
- `key` (optional string) is your id for the line. It is echoed back by `export`, and it is
  the cache key when `isrc` is null. Each line needs at least one of `isrc` and `key`.
- `title`, `artists` and `durationMs` drive the Deezer search fallback, used when the ISRC
  lookup finds nothing or finds a release without a preview. Candidates must match on
  duration ±3 s and on artist name.
- Extra fields are ignored.

### `embed <tracks.jsonl>`

Processes every unique track that is not already cached for the current model. Progress goes
to stderr. One JSON summary line goes to stdout:

```json
{"model": "discogs-effnet-1/mean-hop128/v1", "unique_tracks": 400, "cached": 395, "processed": {"ok": 4, "not_found": 1}, "api_calls": 7, "inference_s": 0.9, "wall_s": 1.8}
```

Options: `--workers 16` (concurrent fetches), `--batch 32` (tracks per inference batch),
`--retry-failed` (ignore retry-after on cached failures), `--quiet`.

### `export <tracks.jsonl>`

Writes one line per input line, in input order, duplicates included:

```json
{"key": "4tKGFmENO69tZR9ahgZu48", "status": "ok", "vector": [0.0123, -0.0441, ...]}
{"key": "0ZsdexTL4uQSBgYHSfRHtt", "status": "no_preview"}
```

- `--prefer-model ID` uses that model id's cached vector where one exists and the default model's elsewhere, e.g. `discogs-effnet-1/mean-hop128/full-yt-v1` from `bench/embed_full.py`.
- `key` is your `key`, or the ISRC when you didn't pass one.
- `vector` is present only when `status` is `ok`. It is 1280 float32 values, L2-normalised,
  so the dot product is cosine similarity.
- `--tags` adds `"tags": {"styles": {"Electronic---Synthwave": 0.36, ...}, "moods": {"energetic": 0.12, ...}}`.
  These are the top 5 Discogs-400 styles and the top 5 MTG-Jamendo moods/themes, with mean
  sigmoid scores. They are for display only.
- `--out FILE` writes to a file instead of stdout.
- `--format npy --out PREFIX` writes `PREFIX.npy`, an `(n_ok, 1280)` float32 array, plus
  `PREFIX.index.jsonl` with `{key, status, row}` per input line. `row` is that line's row in
  the array, or null.

| `status` | Meaning | Retried after |
| --- | --- | --- |
| `ok` | vector available | never (cached for good) |
| `not_found` | neither ISRC nor search found the track on Deezer | 14 days |
| `no_preview` | found, but Deezer has no preview for it | 14 days |
| `decode_error` | preview downloaded but couldn't be decoded | 7 days |
| `download_error` | network/Deezer failure after retries | 1 hour |
| `error` | unexpected exception (bug); message kept in the cache | 1 hour |
| `pending` | (export only) never processed for this model | n/a |

## Library API

```python
from audio_embed_engine import Cache, DiscogsEffnet, embed_tracks, lookup, read_tracks

tracks = read_tracks("tracks.jsonl")            # or Track(isrc=..., title=..., artists=(...), duration_ms=..., key=...)
cache = Cache("embeddings.sqlite")
model = DiscogsEffnet()                         # cheap; weights load on first embed
stats = embed_tracks(tracks, cache, model)      # EmbedStats(total, cached, statuses, api_calls, ...)
for r in lookup(tracks, cache, model.id):       # Result(key, status, vector | None, tags | None)
    ...
```

A different model only has to satisfy the `Model` protocol in `models.py`: an `id`, a
`sample_rate`, and `embed(audios) -> list[Embedding]`. Its rows sit next to the old ones in
the cache, so switching models invalidates nothing.

## Why a CLI, not an HTTP server

The caller (a TypeScript app) needs two things: "make sure these tracks are embedded" and
"give me the vectors". Both are batch operations over a file it already writes. A subprocess
call with JSONL in and JSONL out needs no server lifecycle, no port, and no health checks,
and the SQLite cache makes repeat calls nearly free (~0.2 s for an all-cached run). A local
HTTP API would only pay off for many small interactive requests, and even then the TS side can
read the SQLite file directly. So there's no server.

## How it works

1. **Cache check.** The engine selects every `(track_key, status, retry_after)` row for the
   model in one query (a few ms for thousands of rows). Rows that are `ok`, or failed with a
   retry time still in the future, are skipped. If nothing is left, it returns without any
   network call or model load.
2. **Fetch (16 threads).** Each track is resolved with `GET /track/isrc:<ISRC>`, falling back
   to `/search`. The signed preview URL is downloaded immediately and never stored. Deezer
   API calls go through a sliding-window limiter of 45 per 5 s: the measured limit is ~50 per
   5 s, and going over it returns HTTP 200 with error code 4. Quota errors trigger a full
   window back-off and a retry. Preview downloads go to Deezer's CDN and are not limited.
3. **Decode.** soundfile decodes the bytes in memory. Deezer previews carry an empty ID3 tag
   that libsndfile can't see past, so it is stripped first. soxr then resamples to 16 kHz mono.
4. **Embed (batched, main thread).** The log-mel spectrogram is computed in numpy: a port of
   Essentia's `TensorflowInputMusiCNN`, matched to Essentia within 1e-3 in
   `tests/test_audio.py`. It is cut into non-overlapping 128-frame (~2 s) patches, and
   Discogs-EffNet runs on the whole batch's patches at once. The track vector is the mean
   patch embedding, L2-normalised. Audio is dropped right after this step.
5. **Store.** One row per track: the vector as a float32 blob plus the tags, or a failure
   status with a retry-after time.

## Development

```sh
uv run pytest            # focused tests, no network
uv run ruff check && uv run pyright
```

`bench/` holds the model benchmark. See [bench/REPORT.md](bench/REPORT.md).
