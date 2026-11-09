# Benchmark report (2026-11-06, MacBook Pro M1 Pro)

## Summary

| | |
| --- | --- |
| Model | **Essentia Discogs-EffNet** (`discogs-effnet-bsdynamic-1.onnx`), mean of 1280-d patch embeddings over non-overlapping ~2 s patches, CPU onnxruntime |
| Cache id | `discogs-effnet-1/mean-hop128/v1` |
| Bulk throughput | **~510 tracks/min** end to end (full library: 5,875 unique tracks in ~12 min), bound by Deezer's API quota |
| Inference alone | 37–41 ms/track ≈ 1,500 tracks/min, ~3× faster than the quota lets us download |
| Re-run, all cached | **0.22 s** for 6,233 lines; 0.25 s for the 150-track sample |
| 400 tracks, 5 new | **1.4 s** total (5 API calls, 0.22 s inference) |
| Deezer hit rate | **98.0%** on the 150-track sample, **96.9%** on the whole library (5,690 / 5,875) |
| Same-playlist vs other pairs | mean cosine 0.58 vs 0.44 with same-artist pairs excluded; pair AUC **0.77** (random 0.50) |
| Never-sorted-artist simulation | **recall@3 0.49** on the full library (popularity baseline 0.21, random 0.05) |

## Model choice

The candidates were scored on three things:

- **Speed** on the M1 Pro.
- **Quality** on two label sets:
  - the sorter's 150-track library sample with real playlist memberships, with every
    same-artist pair excluded;
  - a hand-built 53-track fixture: 9 styles × 3 artists × 2 tracks
    (`bench/tracks.jsonl`, built by `bench/build_fixture.py`).
- **Setup pain.**

All models received the same decoded Deezer previews. Metric definitions are in
`bench/metrics.py`:

- **Pair AUC (no shared artist):** the probability that a random same-playlist pair is more
  similar than a random different-playlist pair, after dropping every pair that shares an
  artist. This is the most stable number here, since it is computed over ~10k pairs.
- **LOO recall@3:** leave-one-out. Playlists are ranked for each track using only members
  that share no artist with it. The centroid scorer uses cosine to the mean member; the top-3
  scorer uses the mean of the 3 most similar members. With only 123 queries, differences
  under ~0.05 are noise (standard error ≈ 0.04).
- **Fixture:** a 5-NN style vote and same-style pair AUC, with same-artist neighbours excluded.

Results from `bench/results.json` (`uv run python bench/table.py`):

| model | dim | infer ms/track | pair AUC (no shared artist) | LOO recall@3 centroid | LOO recall@3 top-3 | fixture style 5-NN | fixture style AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **effnet-hop128** (chosen) | 1280 | 41 | **0.769** | 0.264 | **0.310** | 0.74 | **0.963** |
| effnet (hop 62, Essentia default) | 1280 | 80 | 0.769 | 0.249 | 0.278 | 0.74 | 0.963 |
| effnet-coreml | 1280 | 346 | 0.768 | 0.257 | 0.278 | 0.74 | 0.962 |
| discogs-multi | 1280 | 77 | 0.757 | 0.256 | 0.278 | 0.74 | 0.923 |
| effnet-styles400 (400 style probabilities as the vector) | 400 | 79 | 0.755 | 0.256 | 0.265 | 0.75 | 0.919 |
| discogs-artist | 512 | 77 | 0.743 | 0.198 | 0.267 | 0.62 | 0.886 |
| discogs-release | 512 | 78 | 0.729 | 0.240 | 0.275 | 0.62 | 0.893 |
| clap-htsat-unfused | 512 | 82 | 0.679 | 0.224 | 0.293 | 0.57 | 0.882 |
| discogs-label | 512 | 77 | 0.669 | 0.155 | 0.245 | 0.66 | 0.878 |
| musicnn | 200 | 298 | 0.657 | 0.189 | 0.259 | 0.68 | 0.945 |
| discogs-track | 512 | 77 | 0.651 | 0.179 | 0.231 | 0.60 | 0.874 |
| mert (v1-95M, mean over layers and time) | 768 | 405 | 0.607 | 0.198 | 0.251 | 0.38 | 0.737 |
| clap-music (broken, see below) | 512 | 115 | 0.602 | 0.113 | 0.169 | 0.17 | 0.627 |
| random | 256 | 0 | 0.499 | 0.004 | 0.065 | 0.06 | 0.556 |

Why Discogs-EffNet:

- **Quality.** It is best or tied-best on every stable metric. Its training target, 400
  Discogs styles, is close to how people group music into playlists. The Discogs contrastive
  variants (artist, label, release, track) were trained to cluster by exactly those IDs, and
  they generalise *worse* to unseen artists, which is the case that matters here.
- **Speed.** Overlapping patches (Essentia's default hop of 62 frames) double the cost and
  change nothing measurable, so the engine uses hop 128. CoreML is 4× slower than plain CPU
  for this graph, presumably because it falls back for unsupported ops.
- **Setup.** One 18 MB ONNX file plus onnxruntime, which has Apple Silicon wheels. The
  `essentia-tensorflow` Linux-only problem doesn't apply: Essentia ships ONNX exports, and the
  mel front end is reimplemented in numpy. It matches Essentia within 1e-3; see
  `tests/test_audio.py`, where Essentia is only a dev dependency. MERT and CLAP need torch,
  transformers and ~1 GB of weights, and run 2–10× slower here.

Caveats on the losing models. They were run reasonably, but not tuned.

- **MERT** used plain mean-pooling with no probe. Its remote code also ignores
  `output_hidden_states` under transformers 5.22, so layers were captured with forward hooks.
- **`laion/larger_clap_music`** produces collapsed embeddings under transformers 5.22. Mean
  pairwise cosine is 0.94, and zero-shot labels every track "metal". The general
  `clap-htsat-unfused` checkpoint behaves normally and is the fair CLAP number.

**GPU (Gaia, RTX 4060): not run, on purpose.** Inference is ~3× faster than the Deezer quota
lets tracks arrive, so a GPU can't shorten a run. It would only matter for a much heavier
model, and none of the heavier candidates scored better.

## Throughput (full library, default cache)

The first bulk run processed 5,097 tracks in 607 s wall time, about 504 tracks/min, then hit
a bug: a ~1 s preview crashed batch inference. Batches are committed as they finish, so
nothing was lost. After the fix (commit `3d500e0`), the resume processed exactly the
remaining 778 tracks:

```
{"unique_tracks": 5875, "cached": 5097, "processed": {"ok": 773, "decode_error": 1, "no_preview": 3, "not_found": 1},
 "api_calls": 799, "inference_s": 28.43, "wall_s": 89.04}            -> 524 tracks/min
```

How the time splits:

- **Download side.** Each track costs ~1.05 Deezer API calls (the search fallback is rare).
  The limiter allows 45 per 5 s, so ~515 tracks/min is the ceiling, and this is the
  bottleneck.
- **Inference side.** 37 ms/track inside the pipeline (28.4 s for 773 tracks), plus ~28
  ms/track of MP3 decode and resampling on the fetch threads. Both overlap with downloading.
- **Estimate.** 1,500 new tracks ≈ 3 minutes. The model weights (21 MB) download once on
  first use.

Things measured along the way:

- **Deezer API quota.** About 50 calls per 5 s. Going over returns HTTP 200 with
  `{"error":{"code":4,"message":"Quota limit exceeded"}}`, not a 429, and the quota recovers
  ~5 s after the burst. The limiter keeps headroom at 45 per 5 s and backs off a full window
  on code 4. The bulk run hit no quota errors.
- **Preview CDN.** `cdnt-preview.dzcdn.net` sits outside that quota, but answered 429 to
  21 of ~5,000 downloads during the sustained run. Downloads now pause globally on a 429
  (honouring `Retry-After`, else 2–32 s exponential). A `--retry-failed` pass recovered all 21.

Re-run timings (`/usr/bin/time -p uv run audio-embed embed …`):

| run | wall |
| --- | --- |
| 6,233-line library, all cached | 0.22 s (3 runs: 0.22 / 0.22 / 0.22) |
| 150-track sample, fresh cache | 17.6 s |
| 150-track sample, second run | 0.25 s |
| 400 lines (394 unique ISRCs), 5 new | 1.39 s, of which 0.22 s inference, 5 API calls |

## Deezer hit rate

| set | ok | no_preview | not_found | decode_error |
| --- | ---: | ---: | ---: | ---: |
| 150-track sample | 147 (98.0%) | 2 | 1 | 0 |
| full library, 5,875 unique | 5,690 (96.9%) | 107 | 76 | 2 |

The search fallback (artist + title, duration ±3 s, artist-name check) runs rarely: 156 API
calls for the 150 sample tracks, and ~1.05 calls per track across the library. I did not
count how many tracks it rescued, as opposed to confirming a miss. One library entry is a Spotify placeholder with an empty
title, artists and duration, and it is correctly reported as `not_found`.

## Distance check on real playlists

On the 150-track sample (147 embedded), with same-artist pairs excluded:

| pairs | mean cosine | n |
| --- | ---: | ---: |
| share a playlist | **0.581** | 576 |
| don't share a playlist | 0.438 | 9,994 |

The pair AUC is 0.769. Including same-artist pairs moves the mean for pairs sharing a
playlist to 0.606, so the engine isn't leaning on "same artist" much.

### Full library: "artist I've never sorted"

`bench/library_eval.py` uses every one of the 4,617 tracks that sit in a playlist as a
leave-one-out query. Each playlist is scored only from members sharing **no artist** (by
Spotify artist id) with the query, which simulates a song by an artist you've never sorted.
The catch-all "Everything" playlist (5,720 tracks) is ignored, because it is trivially
always correct.

| scorer | recall@3 | hit@3 |
| --- | ---: | ---: |
| mean of 5 most similar members (kNN) | **0.489** | 0.591 |
| cosine to playlist centroid | 0.406 | 0.488 |
| always suggest the 3 biggest playlists | 0.212 | n/a |
| random | 0.046 | n/a |

*recall@3* is the share of a track's playlists that appear in the top 3. *hit@3* is whether
at least one does.

Compare this with the sorter's 13% recall@3 for unseen artists from free signals. The
embedding signal alone covers a large part of that gap. The two numbers aren't directly
comparable, though:

- The sorter's 13% was measured on genuinely unsorted songs. This is a leave-one-out
  simulation over already-sorted songs.
- The sorter's recall@3 definition may differ.

The kNN scorer favours big playlists somewhat, since more members means a higher chance of a
close neighbour. The popularity row shows that this bias alone is worth 0.21, and the
embeddings more than double it. The suggested scorer for the sorter is kNN over playlist
members, after excluding the catch-all playlist.

## Full track vs preview (YouTube Music, 2026-11-06)

Question: does embedding the whole song beat the 30 s Deezer preview? Full tracks came from
YouTube Music (`bench/fetch_youtube.py`), and the comparison was done by `bench/full_vs_preview.py`.
The model was the same and the 193 tracks were the same (150-track sample + fixture).

**Fetching.** The sample was downloaded sequentially from the Mac's residential IP, with 3–8 s
sleeps between tracks.

- **Hit rate:** 195 / 202 unique tracks (96.5%). Most are found through the songs index. ISRC
  search found 3 more, and video search (official audio on artist channels) found 5 more.
- **Failures:** two tracks need sign-in (a per-video gate), and five aren't on YouTube Music
  under a matching artist, title and duration. Covers by other artists are correctly rejected.
- **Bot checks:** none in ~52 minutes.
- **Cost:** ~9 s per track (mostly yt-dlp's JS challenge), plus the sleep, so ~15 s per track.
  Files average 3.6 MB.
- **Server:** from the production server's datacenter IP, YouTube answered
  "Sign in to confirm you're not a bot" for every track tried. This held with and without a JS
  runtime.

**Quality.** All metrics are with shared-artist pairs excluded.

| input | pair AUC | recall@3 top-3 | recall@3 centroid | fixture style AUC | infer ms/track |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deezer preview | 0.769 | 0.281 | 0.274 | 0.963 | 39 |
| full track | 0.760 | 0.298 | 0.269 | 0.958 | 418 |
| middle 30 s of full track | 0.741 | 0.306 | 0.246 | 0.962 | 42 |

- **Pair AUC:** full minus preview is −0.010, with a paired bootstrap 95% CI of
  −0.033 to +0.013. That is no measurable difference on average.
- **How much vectors move:** the median cosine between a track's preview vector and its full
  vector is 0.954, so for most songs the preview already captures the song's average sound.
- **The tail:** the 10% of tracks below 0.87 are correct matches whose preview misses a
  different section (e.g. Bad Bunny's "Después de la Playa", 0.68). In the quarter that moved
  most, the full track placed the track better 8 times and worse 3 times (recall 0.280 → 0.314).
  That fits "the full track helps exactly where the preview is unrepresentative", but n = 29
  can't confirm it. A split this lopsided happens by chance ~23% of the time.
- **What would decide it:** `bench/library_eval.py` on the full library (4,617 queries). That
  needs the whole library downloaded first.

## Full track vs preview, whole library (2026-11-08)

`bench/library_eval.py` leave-one-out on the personal library, scoring playlists only from members
sharing no artist with the query. Both models on the same 5,686 library entries (4,484 queries);
full = whole YouTube Music track via `bench/embed_full.py`, preview = 30 s Deezer. CI is a paired
bootstrap over queries, which treats queries as independent, so it is somewhat optimistic.

| Metric | Preview | Full | Delta (95% CI) |
|---|---|---|---|
| top5 hit@3 | 0.595 | 0.621 | +0.026 (+0.016 .. +0.036) |
| top5 recall@3 | 0.483 | 0.508 | +0.026 (+0.016 .. +0.035) |
| centroid hit@3 | 0.468 | 0.505 | +0.037 (+0.026 .. +0.047) |
| centroid recall@3 | 0.379 | 0.402 | +0.023 (+0.015 .. +0.032) |

Popularity baseline recall@3 is 0.220. Full tracks are a small but consistent gain, unlike the
sample benchmark above, which was too small to detect it. Most queries do not change.

## Not verified

- No GPU benchmark (Gaia), for the reason given above.
- Tags (styles and moods) were checked by eye only. On the fixture they look right:
  - The Midnight → Synth-pop/Synthwave
  - Slipknot → Heavy/Death Metal
  - Nujabes → Instrumental/Jazzy Hip-Hop
  - Daddy Yankee → Reggaeton
  - Yiruma → Neo-Classical
  - Chris Stapleton → Blues Rock (no country in the top 3)

  They are display extras and were not evaluated.
- Long-run CDN behaviour beyond one ~12-minute bulk run.

## Per-playlist sound signal (2026-11-09)

`bench/playlist_signal.py`: 3,500 sorted songs and the 39 playlists the sorter can suggest from sound
(≥12 embedded songs; no hidden, ignored or artist-rule playlists). Vectors are full-track where available.
Five folds by primary artist; training drops every song sharing an artist with a test song. Weights and
calibrators come from an inner split of the training songs only. A whole nested run takes ~8 s on the M1 Pro.

| scorer | recall@3 | hit@3 |
| --- | ---: | ---: |
| linear classifier per playlist (ridge) | **0.668** | 0.711 |
| ridge, Platt-calibrated per playlist | 0.645 | 0.698 |
| mean of 5 most similar members | 0.635 | 0.682 |
| similarity-weighted vote of 15 neighbours (the sorter's audio signal) | 0.614 | 0.665 |
| that vote × per-playlist AUC weight | 0.606 | 0.656 |
| centroid | 0.522 | 0.578 |
| always the 3 biggest playlists | 0.334 | n/a |

Ridge minus vote: +0.054 (95% CI +0.041 to +0.067, bootstrap over songs). The weighted vote minus vote:
−0.008 (−0.016 to +0.002), so a per-playlist multiplier does not help ranking. Per-playlist AUC is useful
as a gate on confident picks instead. Share of picks that are real memberships (a lower bound, since a
missing membership isn't proof a song doesn't fit):

| playlist AUC, from inner folds | P ≥ 0.5 | P ≥ 0.7 | P ≥ 0.9 |
| --- | ---: | ---: | ---: |
| ≥ 0.90 | 0.59 | 0.67 | 0.77 |
| 0.75–0.90 | 0.28 | 0.23 | 0.17 |
| < 0.75 | 0.35 | 0.03 | 0.00 |

Control: with labels shuffled, playlist AUCs run 0.40–0.61 (median 0.49). Track-level PCA: recall@3 for
ridge is 0.668 at 1280-d, 0.662 at 256-d, 0.655 at 128-d; int8 quantisation at 1280-d changes nothing (0.669).

## Classifier tuning (2026-11-09)

`bench/classifier_tuning.py` on the same songs and playlists, 2 × 5 outer folds, with cutoffs chosen only from held-out
scores inside the training songs. **These replace the auto-check precision in the section above:** that table came from one
5-fold split, and this larger test is lower (about 60–67% rather than 67–77%). Results also vary widely by fold (22–95%).

The ridge penalty is flat from 0.3 to 3 (recall@3 0.645, 0.648, 0.649) and worse above it, so it stays at 3. Aiming each
playlist at a target precision directly (lowest score whose held-out picks reach it) undershot: aiming for 70% gave 51–57%.
Picking by a fitted confidence curve did better.

| chance needed | playlist AUC ≥ | only the top playlist | checks per 100 songs | already in the playlist | right per 100 songs |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.5 | 0.90 | yes | 31 | 59% | 18 |
| 0.7 | 0.90 | yes | 22 | 62% | 14 |
| 0.8 | 0.90 | yes | 17 | 64% | 11 |
| **0.9** | **0.90** | **yes** | **12** | **67%** | **8** |
| 0.9 | 0.90 | no | 14 | 65% | 8 |
| 0.9 | 0.93 | yes | 10 | 64% | 7 |

Chosen defaults: chance ≥ 0.9, AUC ≥ 0.90, top playlist only. It has the highest precision of the grid, because a wrong
automatic check costs more than a missed one. Loosening to 0.7 gives more checks at a small precision cost.

In the sorter (`pnpm eval`, songs held out at random), compared with the neighbour vote:

| songs by artists never sorted | vote | classifier |
| --- | ---: | ---: |
| real playlists in top 1 | 30% | 33% |
| top 3 | 55% | 60% |
| top 5 | 65% | 72% |
| pre-checks per song / right | 0 / n/a | 0.05 / 71% (n = 28) |

Songs by known artists keep the same ranking and precheck precision (89%), but only because sound-only picks are limited to
new artists: allowing them for known artists dropped precheck precision to 83%. Playlists mostly made of one album (e.g.
Hamilton) are left out of training, since they score high on shared production and drew 14 of 22 auto-checks in the queue.
