# Do tempo, loudness and vocal features help the playlist classifier?

Script: `bench/classifier_signal.py`. Rerun from the Stemma repo (about 1 minute, CPU; results cached in `~/.cache/audio-embed-engine/classifier-signal/`):

```sh
PYTHONPATH=bench uv run --group bench python bench/classifier_signal.py /Users/axel/Desktop/Code_Projects/Personal/Web/Cento --verify --out /tmp/signal-tables.md
```

## Verdict

**Don't ship.** For songs by artists the classifier has never seen, adding the signal block gives no measurable gain: the best row (all columns at weight 0.25) is +0.3 pp R@3 and +0.2 pp R@1, against a seed-to-seed SD of the paired difference of 0.3 pp and an artist-bootstrap 95% interval of -0.1 to +0.7 pp. At weight 1 or 2 R@3 is exactly flat or slightly negative and R@5 drops (-0.3 to -0.4 pp). Single groups (tempo, loudness or vocals alone) are all within ±0.2 pp. For songs by known artists there is a small gain (+0.4 pp R@3, +0.8 pp R@1, interval +0.1 to +0.7 for R@3, any weight 0.25 to 1), but known-artist songs are already ranked at 84% R@3 by artist-mates in training, and the held-out case is the one the feature was meant to help. Mood and activity playlists (Gym, Chill, Happy, Sad, Flow, Dance) do not move beyond noise: no change above 1.2 pp, each inside its own seed SD, except Happy -0.9 pp (SD 0.6, the only one that looks like a real, slightly negative move, on n=550). If it were ever added anyway, use weight 0.25, which was the best and never harmful, not anything heavier.

## What was measured

- Baseline = sound + language + lyrics@2, the shipped classifier, built through `classifier.with_language` and `classifier.with_block` exactly as `classifier_features.py` does. Same data (3,238 labelled songs, 36 playlists with at least 12 members), same ridge (λ 3), same 5 artist-split folds with artist-mates dropped from training, same fold seeds (`seed + 100`). Seeds 10 to 19. `--verify` checks the baseline's R@3 against the cached lyrics@2 runs of `classifier_features.py` (seeds 0 and 1: 0.7047 and 0.7050, identical to four decimals).
- Signal block (15 columns from `features.jsonl`, plus a has-features column): folded BPM, log2 raw BPM, tempo steadiness, beats-per-bar ≠ 4, integrated loudness, loudness range, intro build time, fade-tail length (log1p) and drop (both 0 when no tail), vocal share, vocal energy, time of first vocal, major/minor, key strength, log duration. Groups for ablation: tempo = first 4, loudness = next 5, vocals = next 3 (the last 3, mode/key strength/duration, only appear in "all"). Standardised on the training fold's songs that have features, missing → 0, then the block is divided by sqrt(#columns) before the weight, so a row is about unit length like the lyrics and sound blocks (so weights are comparable with the earlier report).
- Coverage: features for 93.9% of labelled songs (about 3,040 of 3,238); only `voc_first` has gaps (183 instrumentals or songs with no detected vocals). Songs without features get the has-features column 0 and zeros elsewhere.
- Ablations were run at every weight rather than only at the "best" one, to avoid picking the weight on the result.

## What differs from the earlier report

- Only R@k is computed. The Platt/AUC auto-check part of `classifier_features.py` is skipped (it does not change the ranking), so pick precision and pick recall are not reported. The R@k values for the baseline (71.1% R@3) are on the same scale as the earlier report (70 to 72%).
- "Known artists" is new, not in the earlier protocol: 5 random song folds (not split by artist), training on all other labelled songs, test limited to songs sharing an artist with some training song (2,653 of 3,238). Its baseline numbers (84.3% R@3) are not comparable with the held-out numbers.
- The held-out CI is a bootstrap over artists (1,000 draws, same draw for baseline and variant, per-song values averaged over seeds). Unlike `classifier_features.py`, it is on R@3 only.

## Noise

Held-out baseline R@3 moves ±0.9 pp (SD) between seeds, because each seed is a different artist split, which is three times larger than any gain here. The paired difference to the baseline (same folds) has an SD of 0.1 to 0.4 pp across seeds. Seeds share the same songs, so the 10 seeds do not shrink the uncertainty about the song set; the artist-bootstrap interval is the honest one. A gain smaller than that interval or the paired SD is "no measurable gain", which covers every held-out row. Known-artist R@3 is much steadier (SD 0.2 to 0.3 pp), which is why +0.4 pp shows up there.

## Results

### Held-out artists (artist-split folds, training drops artist-mates)

3238 test songs per seed, 10 seeds (10..19). Mean over seeds; ± is the SD across seeds. Δ is paired with the baseline on the same folds: mean Δ over seeds, SD of Δ across seeds, and a 95% bootstrap interval over artists.

| config | R@1 | R@3 | R@5 | ΔR@1 (pp) | ΔR@3 (pp) | SD of ΔR@3 | 95% CI ΔR@3 | ΔR@5 (pp) |
|---|---|---|---|---|---|---|---|---|
| baseline | 40.0 ± 1.2 | 71.1 ± 0.9 | 81.1 ± 0.8 | - | - | - | - | - |
| signal:all @0.25 | 40.2 ± 1.3 | 71.4 ± 0.9 | 81.2 ± 0.7 | 0.2 | 0.3 | 0.3 | -0.1 to 0.7 | 0.1 |
| signal:all @0.5 | 40.2 ± 1.3 | 71.2 ± 1.0 | 81.0 ± 0.8 | 0.2 | 0.1 | 0.4 | -0.3 to 0.7 | -0.1 |
| signal:all @1 | 40.2 ± 1.4 | 71.1 ± 1.0 | 80.8 ± 0.8 | 0.2 | 0.0 | 0.4 | -0.5 to 0.6 | -0.3 |
| signal:all @2 | 40.2 ± 1.4 | 71.0 ± 1.1 | 80.7 ± 0.8 | 0.2 | -0.0 | 0.4 | -0.6 to 0.6 | -0.4 |
| signal:tempo @0.25 | 39.8 ± 1.2 | 71.2 ± 0.9 | 81.1 ± 0.8 | -0.2 | 0.1 | 0.2 | -0.1 to 0.2 | 0.0 |
| signal:tempo @0.5 | 39.8 ± 1.2 | 71.1 ± 0.9 | 81.1 ± 0.8 | -0.2 | -0.0 | 0.1 | -0.2 to 0.2 | -0.0 |
| signal:tempo @1 | 39.7 ± 1.3 | 71.1 ± 0.9 | 81.0 ± 0.8 | -0.3 | -0.0 | 0.2 | -0.2 to 0.2 | -0.1 |
| signal:tempo @2 | 39.8 ± 1.3 | 71.0 ± 0.9 | 81.0 ± 0.8 | -0.2 | -0.0 | 0.2 | -0.3 to 0.2 | -0.1 |
| signal:loudness @0.25 | 40.1 ± 1.3 | 71.2 ± 0.9 | 81.1 ± 0.7 | 0.1 | 0.1 | 0.3 | -0.1 to 0.3 | 0.0 |
| signal:loudness @0.5 | 40.0 ± 1.3 | 71.1 ± 0.8 | 81.0 ± 0.7 | 0.0 | 0.0 | 0.3 | -0.3 to 0.3 | -0.1 |
| signal:loudness @1 | 40.0 ± 1.3 | 71.0 ± 0.9 | 81.0 ± 0.7 | -0.0 | -0.1 | 0.3 | -0.4 to 0.2 | -0.1 |
| signal:loudness @2 | 40.0 ± 1.4 | 71.0 ± 0.9 | 80.9 ± 0.7 | -0.0 | -0.1 | 0.3 | -0.4 to 0.2 | -0.2 |
| signal:vocals @0.25 | 40.2 ± 1.3 | 71.3 ± 0.9 | 81.1 ± 0.8 | 0.2 | 0.2 | 0.3 | -0.2 to 0.6 | 0.0 |
| signal:vocals @0.5 | 40.2 ± 1.4 | 71.2 ± 1.0 | 81.1 ± 0.8 | 0.2 | 0.2 | 0.3 | -0.3 to 0.6 | -0.0 |
| signal:vocals @1 | 40.1 ± 1.4 | 71.2 ± 1.0 | 81.0 ± 0.8 | 0.1 | 0.1 | 0.4 | -0.4 to 0.6 | -0.1 |
| signal:vocals @2 | 40.1 ± 1.4 | 71.2 ± 1.0 | 81.0 ± 0.8 | 0.1 | 0.1 | 0.4 | -0.3 to 0.6 | -0.1 |

### Known artists (random song folds, artist-mates stay in training)

2653 test songs per seed, 10 seeds (10..19). Mean over seeds; ± is the SD across seeds. Δ is paired with the baseline on the same folds: mean Δ over seeds, SD of Δ across seeds, and a 95% bootstrap interval over artists.

| config | R@1 | R@3 | R@5 | ΔR@1 (pp) | ΔR@3 (pp) | SD of ΔR@3 | 95% CI ΔR@3 | ΔR@5 (pp) |
|---|---|---|---|---|---|---|---|---|
| baseline | 57.0 ± 0.2 | 84.3 ± 0.3 | 90.9 ± 0.2 | - | - | - | - | - |
| signal:all @0.25 | 57.6 ± 0.2 | 84.7 ± 0.2 | 91.2 ± 0.3 | 0.6 | 0.4 | 0.3 | 0.2 to 0.7 | 0.2 |
| signal:all @0.5 | 57.8 ± 0.2 | 84.7 ± 0.1 | 91.1 ± 0.3 | 0.8 | 0.4 | 0.3 | 0.1 to 0.8 | 0.2 |
| signal:all @1 | 57.8 ± 0.3 | 84.7 ± 0.2 | 91.1 ± 0.3 | 0.8 | 0.4 | 0.3 | 0.1 to 0.7 | 0.2 |
| signal:all @2 | 57.8 ± 0.2 | 84.6 ± 0.2 | 91.1 ± 0.3 | 0.8 | 0.3 | 0.3 | 0.0 to 0.7 | 0.1 |
| signal:tempo @0.25 | 57.0 ± 0.2 | 84.4 ± 0.2 | 90.9 ± 0.3 | 0.0 | 0.1 | 0.2 | 0.0 to 0.3 | -0.1 |
| signal:tempo @0.5 | 57.0 ± 0.2 | 84.5 ± 0.2 | 90.8 ± 0.3 | 0.0 | 0.2 | 0.2 | 0.0 to 0.3 | -0.1 |
| signal:tempo @1 | 57.1 ± 0.2 | 84.5 ± 0.3 | 90.8 ± 0.3 | 0.1 | 0.2 | 0.1 | 0.1 to 0.3 | -0.1 |
| signal:tempo @2 | 57.0 ± 0.2 | 84.5 ± 0.3 | 90.8 ± 0.3 | 0.0 | 0.2 | 0.2 | 0.0 to 0.3 | -0.1 |
| signal:loudness @0.25 | 57.4 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.3 | 0.4 | 0.2 | 0.2 | -0.1 to 0.4 | -0.0 |
| signal:loudness @0.5 | 57.4 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.2 | 0.4 | 0.2 | 0.2 | -0.1 to 0.4 | -0.0 |
| signal:loudness @1 | 57.4 ± 0.2 | 84.5 ± 0.3 | 90.9 ± 0.2 | 0.4 | 0.2 | 0.2 | -0.1 to 0.4 | -0.0 |
| signal:loudness @2 | 57.3 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.2 | 0.3 | 0.2 | 0.2 | -0.1 to 0.4 | -0.0 |
| signal:vocals @0.25 | 57.4 ± 0.2 | 84.4 ± 0.2 | 91.0 ± 0.2 | 0.4 | 0.1 | 0.2 | -0.0 to 0.4 | 0.1 |
| signal:vocals @0.5 | 57.5 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.2 | 0.5 | 0.1 | 0.2 | -0.1 to 0.4 | 0.0 |
| signal:vocals @1 | 57.5 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.2 | 0.5 | 0.1 | 0.2 | -0.1 to 0.3 | -0.0 |
| signal:vocals @2 | 57.4 ± 0.2 | 84.4 ± 0.3 | 90.9 ± 0.2 | 0.4 | 0.1 | 0.2 | -0.1 to 0.3 | -0.0 |

### Per playlist, held-out artists: baseline → `signal:all @0.25` (the highest held-out R@3 config)

Playlist recall@3 = share of the playlist's songs (as test songs) that have the playlist in their top 3. Seed-averaged. SD = spread of the change across seeds. These are small groups: a change under ~5 pp is within noise.

| playlist | n | before → after (%) | Δ pp | SD of Δ |
|---|---|---|---|---|
| BEST SONIC SONGS | 173 | 72.4 → 77.6 | +5.3 | 1.2 |
| J-Rock | 22 | 28.2 → 31.4 | +3.2 | 4.8 |
| Lovey Dovey | 66 | 40.2 → 42.0 | +1.8 | 2.7 |
| Random uwu | 64 | 16.9 → 18.6 | +1.7 | 2.5 |
| Psych | 86 | 16.4 → 17.9 | +1.5 | 2.8 |
| Vibey Dance | 33 | 26.1 → 27.3 | +1.2 | 5.6 |
| Happy :) | 550 | 80.5 → 79.5 | -0.9 | 0.6 |
| Chill | 75 | 7.5 → 8.4 | +0.9 | 1.3 |
| Regueton | 44 | 72.5 → 73.4 | +0.9 | 2.7 |
| Deep Dive | 67 | 15.5 → 14.6 | -0.9 | 2.5 |

Mood/activity playlists (matched by name; not selected on the result), same config:

| playlist | n | before → after (%) | Δ pp | SD of Δ |
|---|---|---|---|---|
| Chill | 75 | 7.5 → 8.4 | +0.9 | 1.3 |
| Flow | 176 | 85.6 → 85.5 | -0.1 | 1.2 |
| Chillhop | 15 | 24.0 → 24.0 | +0.0 | 3.1 |
| Sad :( | 247 | 86.8 → 86.7 | -0.2 | 0.3 |
| Gym | 17 | 0.0 → 0.0 | +0.0 | 0.0 |
| Vibey Dance | 33 | 26.1 → 27.3 | +1.2 | 5.6 |
| Flow >:D | 101 | 29.9 → 30.3 | +0.4 | 1.7 |
| Dance | 18 | 0.6 → 0.6 | +0.0 | 0.0 |
| Chill Breakcore | 43 | 88.4 → 87.7 | -0.7 | 1.6 |
| Happy :) | 550 | 80.5 → 79.5 | -0.9 | 0.6 |

The per-playlist tables are for `signal:all @0.25`. The "10 that move most" table is selected on the size of the change, so it overstates real effects: the biggest mover (BEST SONIC SONGS, +5.3 pp, SD 1.2, n=173) is the only one that clears its own seed SD, and it is not a mood playlist. I have not tested whether it is a real effect. The mood/activity table is not selected on results. Gym (n=17) and Dance (n=18) sit at about 0% before and after, meaning the classifier cannot place their new-artist songs in the top 3 with or without these features.

## Inferences (not measured)

- Tempo and loudness may be already carried by the 1280-d sound vector, so a separate column adds little. I did not test this.
- The small known-artist gain probably comes from songs where artist-mates in training do not settle the playlist; I did not look at which songs.

## Did not run

- No λ retune for the signal block. A weight changes the effective penalty, so some of the weight effect is a penalty effect.
- Not run through Cento's `pnpm eval`, no auto-check precision/recall, no nonlinear use of the features (a ridge on standardised columns cannot express "BPM near 170" or an interaction between tempo and vocals).
- One song set; the 5 songs whose features extraction failed are treated as having none.
