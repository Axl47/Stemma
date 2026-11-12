# Personal metric: does a learned transform of the vectors match the listener's sorting better?

Run with `PYTHONPATH=bench uv run --group bench python bench/personal_metric.py <cento>/data`.
The numbers below are from 2026-11-12: 5,783 embedded songs, 3,252 of them in one of the 43 counted
playlists (`classifier-input.jsonl`: no all-songs, ignored, hidden, artist-rule or single-album playlists;
sizes 2 to 550, 21 of them under 40 songs). No playlist names or song data appear here.

## Conclusion

**It helps in one specific way, and not in the way the goal hoped.** A transform trained on playlist
co-membership makes a song's nearest neighbours and its playlist suggestions better *for playlists that
existed at training time*. It does not make the whole space agree better with the listener's sorting, and
it does nothing for a playlist created after training.

- **Global agreement (pair AUC): no effect.** Best gain +0.017 (contrastive) against a fold-to-fold noise
  of about ±0.03 per fold (±0.014 as a standard error). Baseline is 0.738.
- **Nearest neighbours (nn15): a real, consistent gain.** 0.341 to 0.41 (+0.07 ± 0.02 paired over 15 folds),
  and it holds after subtracting each song's own base rate of sharing a playlist (0.262 to 0.33).
- **Placing a song (recall@3): a small gain of +0.02** (0.648 to 0.668), real but modest.
- **Small playlists (under 40 songs) gain the most, not the least.** recall@3 on them goes 0.18 to 0.30
  (+0.12 ± 0.04); nn15 on them 0.063 to 0.09. The suggestions lean *less* on the big playlists
  (share of top-3 picks that are the 3 biggest playlists: 0.30 to 0.23), so this is not collapse.
- **Unseen playlists: no gain.** With whole playlists held out of training, every transform lands inside
  noise of baseline on pair AUC (−0.006 to +0.009) and slightly *below* it on neighbours (−0.01). The same
  playlists, when seen in training, gain +0.04 to +0.05 AUC and +0.04 to +0.07 nn15. So what is learned is the
  listener's existing playlists, not a general notion of "how this listener files things".
- **Whitening alone hurts.** Full whitening cuts pair AUC by 0.07 to 0.14. Centring or plain PCA is neutral
  on neighbours and costs 0.02 AUC.
- **A small MLP is worse** than the linear versions on every headline metric, so it is not worth pursuing.
- **Dimension barely matters** across 32 to 256 (blend: 68, 100, 164 and 236 dims are within 0.001 of each
  other to within 0.003 on every metric). Use the smallest.

The gains in nn15 and small-playlist recall are exactly what the existing per-playlist classifier
(`classifier.json`) already encodes. A transform would let neighbours, the map and bridges inherit that
knowledge, but it will not discover new categories, and it needs retraining whenever playlists change.

## Method

- **Folds.** 5 folds by primary artist, repeated for 3 shuffles (15 folds). Test songs are the fold. Training
  songs are every other song sharing no artist with any test song (asserted in code). Whole connected groups
  could not be kept together: the featuring graph joins 3,182 of 5,783 songs into one group, so dropping every
  training song that touches a test artist is used instead; it gives the same no-shared-artist guarantee.
  Nothing (mean, PCA, regression, network) is fitted on test songs. Unsupervised fits (mean, PCA) may use
  unlabelled songs of non-test artists; supervised fits use labelled training songs only.
- **Metrics.** Pair AUC (shared-artist pairs dropped; verified equal to `metrics.pair_auc` on a subset),
  recall@3 with the `library_eval.py` top-5-members scorer (centroid scorer in the JSON output), and nn15:
  share of a test song's 15 nearest training songs that share a playlist with it. "Small" columns use only
  songs in playlists under 40 songs: recall@3 against their small playlists, nn15 against shared small
  playlists, pair AUC with positives = pairs sharing a small playlist.
- **Big and small playlists.** Pair AUC and nn15 are song-level, so big playlists dominate them; hence the
  separate small-playlist columns. Supervised fits skip playlists with fewer than 5 training members in the
  fold. "Balanced" variants standardise each playlist's target (ridge) or weight anchors by
  1/sqrt(size of the song's smallest playlist) (contrastive). Balancing trades about 0.01 pair AUC and a
  little recall for more small-playlist recall (+0.03 to +0.04 for contrastive).
- **Noise.** Every cell is mean ± fold sd. Deltas are paired against baseline on the same folds, and a gain
  counts only if it exceeds 2 sd/sqrt(5). The 3 shuffles reuse the same songs, so the effective sample is
  nearer 5 than 15.
- **Hyperparameter choice.** Sweeps are shown whole. The contrastive settings (60 whole-batch steps, tau 0.3,
  PCA init) were picked on a separate dev split (seed 7) of the same songs; the other grids were not tuned.
  There is no nested selection, so the best rows are slightly optimistic. None of the findings depend on
  that, since the interesting gain is far outside the noise and the null results stay null.

## Results

Mean ± sd over 15 folds. Test-fold songs only, artists never seen in training. Baseline row first; the
`Δ` column in the text above is paired. "pop3" is the share of top-3 suggestions that are the 3 biggest
playlists (lower means less leaning on big playlists; always suggesting those three scores recall@3 0.226).

| transform | dim | pair AUC | recall@3 | nn15 | AUC small | recall@3 small | nn15 small | pop3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline cosine** | 1280 | 0.738±0.060 | 0.648±0.034 | 0.341±0.031 | 0.789±0.069 | 0.180±0.050 | 0.063±0.013 | 0.303 |
| mean-centred | 1280 | 0.716±0.089 | 0.655±0.033 | 0.348±0.032 | 0.803±0.070 | 0.203±0.047 | 0.068±0.012 | 0.280 |
| PCA (no whitening) | 128 | 0.717±0.089 | 0.656±0.031 | 0.349±0.033 | 0.803±0.070 | 0.208±0.048 | 0.069±0.011 | 0.278 |
| PCA, full whitening | 32 | 0.671±0.084 | 0.639±0.036 | 0.334±0.033 | 0.720±0.062 | 0.189±0.042 | 0.064±0.010 | 0.283 |
| PCA, full whitening | 256 | 0.595±0.060 | 0.646±0.038 | 0.318±0.030 | 0.629±0.050 | 0.196±0.050 | 0.063±0.011 | 0.303 |
| ridge to playlist matrix, λ=10, standardised | 36 | 0.749±0.089 | 0.669±0.036 | 0.389±0.037 | 0.808±0.058 | 0.238±0.050 | 0.075±0.013 | 0.241 |
| ridge, λ=1, standardised | 36 | 0.744±0.083 | 0.666±0.032 | 0.424±0.039 | 0.797±0.065 | 0.306±0.060 | 0.098±0.017 | 0.224 |
| ridge, λ=0.1, standardised | 36 | 0.700±0.077 | 0.623±0.036 | 0.436±0.042 | 0.750±0.063 | 0.336±0.064 | 0.118±0.021 | 0.211 |
| **blend**: PCA-64 (α=0.5) 30% + ridge 70% | 100 | 0.745±0.084 | 0.668±0.033 | 0.408±0.037 | 0.802±0.067 | 0.297±0.058 | 0.089±0.014 | 0.230 |
| blend, PCA-32 | 68 | 0.745±0.085 | 0.668±0.033 | 0.411±0.037 | 0.805±0.066 | 0.291±0.058 | 0.091±0.014 | 0.230 |
| blend, PCA-200 | 236 | 0.744±0.084 | 0.669±0.034 | 0.409±0.038 | 0.801±0.066 | 0.299±0.056 | 0.091±0.014 | 0.228 |
| contrastive, plain | 32 | 0.755±0.080 | 0.655±0.033 | 0.410±0.047 | 0.789±0.057 | 0.244±0.061 | 0.074±0.012 | 0.225 |
| contrastive, balanced | 64 | 0.744±0.079 | 0.645±0.035 | 0.402±0.046 | 0.792±0.057 | 0.284±0.068 | 0.085±0.016 | 0.220 |
| contrastive, plain | 256 | 0.751±0.076 | 0.639±0.030 | 0.419±0.049 | 0.770±0.056 | 0.239±0.052 | 0.066±0.012 | 0.214 |
| MLP head (hidden 256) | 64 | 0.730±0.073 | 0.607±0.031 | 0.389±0.046 | 0.737±0.046 | 0.220±0.056 | 0.063±0.014 | 0.222 |

Paired gains over baseline (mean ± sd of per-fold difference; noise threshold on AUC about 0.027):

| transform | Δ pair AUC | Δ recall@3 | Δ nn15 | Δ recall@3 small |
|---|---:|---:|---:|---:|
| ridge λ=10 std | +0.011 ± 0.034 | +0.022 ± 0.019 | +0.048 ± 0.017 | +0.058 ± 0.038 |
| blend PCA-64 + ridge (70%) | +0.006 ± 0.030 | +0.020 ± 0.014 | +0.067 ± 0.018 | +0.116 ± 0.041 |
| contrastive d=32 plain | +0.017 ± 0.030 | +0.007 ± 0.013 | +0.069 ± 0.025 | +0.064 ± 0.053 |
| contrastive d=64 balanced | +0.006 ± 0.031 | −0.003 ± 0.017 | +0.061 ± 0.025 | +0.103 ± 0.048 |
| MLP head | −0.009 ± 0.031 | −0.040 ± 0.018 | +0.048 ± 0.024 | +0.040 ± 0.041 |

### Playlists never seen in training

Playlists with at least 8 songs (about 30) were split into 5 groups; for each group, the transform is trained
without those playlists' labels and scored only on pairs/neighbours defined by them (pairs sharing only held-out
playlists vs pairs sharing none). 2 shuffles × 5 folds × 5 groups. A song that is also in a non-held-out playlist
keeps that label in training.

| transform | pair AUC, unseen | pair AUC, same playlists seen | nn15, unseen | nn15, seen |
|---|---:|---:|---:|---:|
| baseline cosine | 0.727±0.091 | n/a | 0.279±0.094 | n/a |
| PCA-32 (unsupervised) | 0.731 (+0.005±0.050) | n/a | 0.287 (+0.008±0.019) | n/a |
| ridge λ=10 std, rank 16 | 0.735 (+0.009±0.058) | 0.773 (+0.046±0.062) | 0.265 (−0.015±0.032) | 0.319 (+0.040±0.032) |
| blend PCA-32 + ridge | 0.723 (−0.004±0.052) | 0.766 (+0.039±0.063) | 0.269 (−0.011±0.030) | 0.345 (+0.066±0.039) |
| contrastive d=32 plain | 0.720 (−0.006±0.051) | 0.773 (+0.047±0.058) | 0.266 (−0.014±0.030) | 0.330 (+0.051±0.044) |
| MLP head | 0.663 (−0.064±0.046) | 0.744 (+0.018±0.058) | 0.216 (−0.063±0.041) | 0.316 (+0.036±0.051) |

Deltas are against the baseline's unseen score. The gain exists only where the label was available.

## Reading the numbers

- **Fact:** gains on neighbours and small-playlist placement are well outside noise; gains on pair AUC are not;
  gains on held-out playlists are absent.
- **Inference:** the transform sharpens the directions that separate the playlists it was given (it is close to
  a classifier), which cleans up *local* neighbourhoods but does not reorder far-apart pairs, hence nn15 moves and
  pair AUC does not. I did not test this directly.
- **Not checked:** whether the gain survives playlists drifting over time (all folds mix old and new songs), and
  any effect on the 2-D map or bridges (only similarity metrics were measured, not those features).

## If it ships

Only worth it if the goal is "neighbours and placement for the current playlists", and only the cheap version:

- **Winner:** the blend: PCA-64 (α=0.5, unsupervised, mean-centred) plus the standardised ridge to the
  membership matrix (λ=1 × mean eigenvalue), similarity = 0.3 · cos(PCA part) + 0.7 · cos(ridge part). It matches
  or beats contrastive on every metric except pair AUC (inside noise), is deterministic, and fits in
  ~0.1 s (0.37 s with the PCA eigendecomposition) on all 3,216 training songs. Contrastive needs ~4 s and torch.
- **Format:** one `float32` matrix `1280 × (64 + P)` (about 0.5 MB for P ≈ 39) plus the mean vector, the
  block split and the 0.3/0.7 weights, written by the engine next to `classifier.json`. Cento computes
  `y = (x − mean) · W`, L2-normalises each block, scales by √weight and concatenates; the dot product of two
  results is the blended cosine, so nearest-neighbour code is unchanged.
- **Retraining:** in the daily refresh, right after `classify`, since the labels come from the same
  `classifier-input.jsonl`. Under a second, so it adds nothing noticeable.
- **Limits to state in the UI or code:** a new playlist gets no benefit until the next retrain *and* has enough
  songs (at least 5 in training); until then similarity for it is plain baseline. Keep the raw vectors: the
  playlist-suggestion path should keep using the unmodified cosine unless this is evaluated there separately,
  since recall@3 only moves by +0.02.
- **Cheaper alternative worth weighing:** reuse the existing `classifier.json` scores as extra features for
  neighbours instead of a new matrix; this is nearly the same model.
