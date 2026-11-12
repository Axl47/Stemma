# Does the classifier place new-artist songs better with more features?

Script: `bench/classifier_features.py`. Run from the Stemma repo:
`uv run --no-sync python bench/classifier_features.py <cento repo> --seeds 10 11 12 13 14 15 16 17 18 19 --sets +vocals +lyrics lyrics:2+vocals:0.5 ...`
(BLAS capped at 4 threads; each (feature set, seed) result is cached in `~/.cache/audio-embed-engine/classifier-features/`.)

## Short answer

Two features help, a little. Nothing here changes the picture that sound alone is weak on artists you've never sorted.

- **Lyric passage vectors** (mean of a song's multilingual-e5 passage vectors) help most: about +1.2 pp recall@3.
- **Vocal stem vector** helps a bit: about +0.8 pp recall@3.
- **Both together: R@3 69.8% → 71.8% (+2.0 pp, 95% interval +1.0 to +3.2)**, R@1 +1.5 pp, R@5 +1.5 pp, and the auto-check precision goes from 76.7% to 80.8% with 12% more auto-checks.
- Lyric theme scores, the instrumental ("rest") stem, and per-moment std or max add nothing measurable. The "rest" vector, and heavier weights on most things, make it worse.

The gain is real but modest (2 more of every 100 true playlists land in a song's top 3). It is not going to move auto-check recall out of single digits.

## Protocol

Same as `bench/classifier_tuning.py` and the shipped `classifier.train`, run on the real Cento inputs (`classifier-input.jsonl` + `vectors.jsonl`, 3,238 labelled songs, 36 playlists, the set `pnpm classify` trains on):

- 5 folds split by primary artist; training songs sharing any artist with a test song are dropped.
- Ridge (lambda 3) per playlist. Platt curves and the playlist AUC trust are fitted on held-out scores *inside* the training songs (2 × 5 inner artist folds), then everything is judged on the test fold with the shipped rule: confidence 0.9, min AUC 0.9, at most one auto-check per song.
- R@k = share of a song's true playlists inside its top k by classifier score, averaged over songs. Pick precision = auto-checks that are right / auto-checks. Pick recall = per song, right auto-check / number of true playlists, averaged (Cento's definition).
- **Leakage guard.** Nothing is fitted on a test fold's songs or artists: the language columns, and the stand-in for a missing block (the mean of that block over the fold's *training* songs), come from training rows only. Blocks are normalised per row or by a constant, so there is no fitted scaler. There is no PCA.
- **Missing features.** A song without a block gets that block's training mean, which is zero after the ridge's centring, so it is scored from the remaining blocks. Every song stays scorable.
- Each block is appended to the baseline vector (mix + language columns) scaled by a weight. Stems, lyric vectors and patch stats are L2-normalised per row; theme scores are divided by sqrt(40).

Comparison to Cento's `pnpm eval`: its "new" column (R@3 62%) is a different test (random song folds, only songs whose artist has no training songs, scored through the full suggestion model), so the absolute numbers are not comparable. The baseline here reads 69.8% R@3, 39.0% R@1. Only the deltas matter.

### Coverage

| block | songs with it | of labelled songs |
|---|---|---|
| vocals, rest, patch std/max | 5,417 of 5,783 (93.7%) | 3,043 of 3,238 (94.0%) |
| themes, lyrics | 4,881 (84.4%) | 2,867 (88.5%) |
| lyrics + vocals together | | 2,731 (84.3%) |

Unlabelled songs (the ones the app actually scores) have the same coverage: 93% stems, 79% lyrics. I did not check why the rest are missing (likely no YouTube match for stems, and instrumentals or no lyrics found for lyrics).

### Noise

The baseline R@3 swings **4.8 pp (SD) from one fold to the next** (10 seeds × 5 folds), which is far larger than any gain. So deltas are judged **paired**: the same folds and test songs for baseline and variant. The "paired fold Δ" SD column (0.7 to 1.7 pp) is the spread of that difference per fold, and the interval is a 95% bootstrap over whole artists (1,000 draws, same draw for both arms). A gain counts only when its interval excludes zero; otherwise "no effect". Intervals cover test-set sampling, not the choice of weights (see below) or the 22+ comparisons made, so a delta only just clear of zero should be read as a hint.

## Results

### Confirmation run: fresh seeds 10 to 19 (50 folds), baseline vs each feature set

Rows at `@1` use an a priori weight of 1 for every block (no tuning). The three bottom rows use the weights picked from the screening run below (seeds 0 to 4); the seeds here are different artist splits of the same songs.

| feature set | R@1 | R@3 | R@5 | pick precision | pick recall | auto-checks/song | ΔR@3 vs baseline (95% interval) | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 39.0% | 69.8% | 80.0% | 76.7% | 6.5% | 0.112 | - | - |
| +vocals @1 | 39.9% | 70.7% | 80.4% | 76.2% | 7.3% | 0.125 | +0.9 pp (-0.1 to +2.2) | no effect |
| +rest @1 | 39.1% | 69.5% | 79.7% | 74.8% | 6.3% | 0.112 | -0.4 pp (-0.9 to +0.2) | no effect |
| +vocals+rest @1 | 40.2% | 70.5% | 80.1% | 75.5% | 7.2% | 0.124 | +0.7 pp (-0.4 to +2.0) | no effect |
| +themes @1 | 39.0% | 69.6% | 79.8% | 77.4% | 6.8% | 0.117 | -0.3 pp (-0.8 to +0.2) | no effect |
| +lyrics vectors @1 | 39.8% | 70.6% | 80.9% | 79.1% | 6.9% | 0.115 | +0.7 pp (+0.4 to +1.1) | real, small |
| +patch std @1 | 39.3% | 69.9% | 80.2% | 76.0% | 6.7% | 0.117 | +0.1 pp (-0.2 to +0.5) | no effect |
| +patch max @1 | 39.3% | 70.0% | 80.1% | 76.3% | 6.7% | 0.117 | +0.1 pp (-0.2 to +0.5) | no effect |
| +vocals @0.5 | 39.8% | 70.7% | 80.5% | 76.8% | 7.0% | 0.120 | +0.8 pp (+0.2 to +1.8) | real, small |
| +lyrics vectors @2 | 40.0% | 71.1% | 81.1% | 80.7% | 7.1% | 0.116 | +1.2 pp (+0.7 to +1.8) | real, small |
| **+lyrics @2 +vocals @0.5** | **40.5%** | **71.8%** | **81.5%** | **80.8%** | **7.8%** | **0.125** | **+2.0 pp (+1.0 to +3.2)** | **real** |

Best combination against baseline on all six numbers (artist bootstrap, same draws):

| metric | baseline | combination | Δ (95% interval) |
|---|---|---|---|
| R@1 | 39.0% | 40.5% | +1.5 pp (+0.6 to +2.2) |
| R@3 | 69.8% | 71.8% | +2.0 pp (+1.0 to +3.2) |
| R@5 | 80.0% | 81.5% | +1.5 pp (+0.7 to +2.4) |
| auto-check precision | 76.7% | 80.8% | +4.1 pp (+2.0 to +6.0) |
| auto-check recall | 6.5% | 7.8% | +1.3 pp (+0.6 to +2.0) |
| auto-checks per song | 0.112 | 0.125 | +0.013 (+0.005 to +0.021) |

The combination beats the baseline in 46 of 50 folds. About 400 auto-checks per pass are made over the 3,238 songs, so the precision gain rests on few events; its interval is the honest guide, not the point estimate.

Auto-check precision is about 77% to 81% on held-out artists even though the rule asks for 0.9. That is the existing behaviour, not something these features cause or fix. Part of it is that a playlist a song is missing from may still fit it (the tuning script calls "real" a lower bound).

### Screening run: seeds 0 to 4 (25 folds), weight sweep

Baseline R@3 70.0%, fold SD 5.9 pp. Δ is R@3 vs baseline, with 95% interval.

| block | @0.5 | @1 | @2 | @3 | @4 |
|---|---|---|---|---|---|
| vocals | +0.8 (+0.1 to +1.5) | +0.7 (-0.4 to +1.7) | +0.1 (-1.3 to +1.5) | | |
| rest | +0.1 (-0.2 to +0.4) | -0.6 (-1.1 to -0.1) | -2.4 (-3.4 to -1.4) | | |
| vocals+rest | +0.6 (-0.0 to +1.4) | +0.5 (-0.6 to +1.6) | -1.1 (-2.5 to +0.4) | | |
| themes | +0.3 (-0.2 to +0.8) | +0.0 (-0.6 to +0.5) | -0.2 (-0.8 to +0.4) | | |
| lyrics vectors | +0.4 (+0.1 to +0.6) | +0.9 (+0.5 to +1.4) | +1.5 (+0.8 to +2.1) | +1.1 (+0.2 to +1.9) | +0.6 (-0.4 to +1.6) |
| patch std | +0.1 (-0.1 to +0.3) | +0.1 (-0.3 to +0.4) | -0.7 (-1.4 to -0.0) | | |
| patch max | +0.0 (-0.1 to +0.2) | +0.1 (-0.3 to +0.4) | -0.8 (-1.5 to -0.1) | | |

Combinations (screening seeds): lyrics@2 + vocals@0.5 +1.9 pp (+0.9 to +2.9), lyrics@3 + vocals@0.5 +1.5, lyrics@3 + vocals@0.5 + themes@0.5 +1.3, lyrics@3 + themes@1 +0.7 (noise). Adding themes never helped.

**Selection bias.** The weights and the winning combination were picked on the screening seeds and then confirmed on fresh artist splits of the *same songs*. That removes fold-split luck but not the fact that the songs are the same, so treat +2.0 pp as slightly optimistic. The a priori rows (`@1`) are not selected on anything.

## Conclusion

- **Lyric passage vectors help** (+0.7 pp at weight 1, +1.2 pp at weight 2, on the 89% of songs that have lyrics; songs without them are unaffected). It also makes auto-checks more precise (79% to 81% vs 77%). Too large a weight (4) loses the gain.
- **The vocal stem vector helps slightly** (+0.8 pp at weight 0.5, 94% coverage) and only at a low weight. It adds auto-checks without hurting precision.
- **Together +2.0 pp R@3**, partly additive since they carry different information (words vs sung timbre).
- **No effect:** lyric theme scores (probably because the passage vectors already carry that information; not tested), the instrumental "rest" stem (it hurts at weight 1 and above), and per-moment std or max (`@2` hurts, `@1` is exactly zero; the song mean already holds what the model can use).
- Whether this is worth shipping is a judgement call: +2 pp R@3 and +4 pp auto-check precision, but auto-check recall is still only ~8% and new-artist R@1 is 40%. It stays far below what an artist rule or a known-artist neighbour gives.

## What shipping the winner takes

**Shipped: lyrics only** (weight 2, `classify --lyrics <per-song vectors.jsonl>`, via `classifier.with_block`; this script now builds its design matrix through the same function, and its lyrics@2 rows reproduce with it). The vocals block and the `--lyrics PATH` index format sketched below were not shipped.

Weights: lyrics 2.0, vocals 0.5; missing blocks mean-filled from the labelled songs (the training mean).

Stemma (`audio-embed classify`, shipped in `src/audio_embed_engine/`):
- `classifier.py`: a `with_blocks(x, blocks)` step after `with_language` that appends weighted blocks and fills a missing song's block with the mean of the labelled songs. `train`'s held-out scoring should recompute the fill from each fold's training songs, as `Data.design` does here.
- `cli.py`: new options, e.g. `--stems DIR` (reads `<key>.npz['vocals']`, keyed by `isrc`) and `--lyrics PATH` (`lyrics-index.json`, `.bin` next to it, mean per song, keyed by ISRC or Spotify id). Both optional; absent means the current behaviour.
- `README.md` (classify section) and a focused test for the missing-block fill.

Cento:
- `src/cli/classify.ts`: write `isrc` into `data/classifier-input.jsonl` rows (so the engine can find a song's stem and lyric rows; the file today only has the Spotify id) and add `--stems ~/.cache/audio-embed-engine/stem-vectors --lyrics data/lyrics-index.json` to the engine command (`ENGINE_CLASSIFY` default).
- The data must exist where it runs: `data/lyrics-index.json` and `.bin` already come from `pnpm lyrics`. The stem vectors are on the storage box and I copied them by hand; I found no Cento step that produces them for newly liked songs, so a new song is scored without the vocals block until `bench/stems_full.py` has run for it (the mean-fill makes that safe). The `pnpm host` refresh would need to run that step to keep the vocals block fresh. Dropping vocals and shipping lyrics only costs about 0.8 pp of the 2.0 pp and needs no new data flow.
- No change to `src/core/classifier.ts` or the shape of `classifier.json`.

## Did not run

- Not run through Cento's own `pnpm eval` (it reads the shipped `data/classifier.json`, which I must not overwrite), so the effect on the final suggestion ranking and the "new" column is not measured. The deltas above are the classifier alone.
- No lambda retune per feature set (fixed at 3.0); a block weight changes the effective penalty, so some of the weight effect is a penalty effect.
- One song set (this library). The weights are not nested-cross-validated; see the selection-bias note.
- Auto-checks on songs by *known* artists, which the app handles through other signals, were not examined.
- Only the final classification metrics were checked; no per-playlist breakdown.
