# Per-song features: tempo, key, loudness, vocals

`bench/extract_features.py` turns each full track on the storage box into one JSON line of the things a playlist builder needs to order songs by hand-off: tempo and beats, musical key, loudness over time, and when vocals start and stop. Audio stays on the box: every batch of mixes and vocals stems is rsynced to a temp dir (the next batch downloads while this one is processed) and deleted afterwards, as in `embed_full.py`. Nothing is sent to an outside service.

## Tools

Checked to install and run together in the script's own environment (Python 3.12, macOS arm64): beat-this 1.1.0 (model `final0`), essentia 2.1b6.dev1389 (cp312 wheel; `KeyExtractor` works), torch 2.15.0 with mps available, numpy 2.5.4, soxr. ffmpeg 9.0.1 was already installed (Homebrew); nothing was installed system-wide.

## Run

The script carries its own dependencies (PEP 723, Python 3.12, resolved as of 2026-11-12), so it needs neither the project's virtualenv nor `uv sync`:

```sh
cd Stemma
F=~/.cache/audio-embed-engine/features
rsync -t audio-box:data/sources.jsonl $F/                 # the box's list of songs and files

# whole library (resumable; stop with ctrl-C or kill, rerun the same command)
uv run --no-project --python 3.12 bench/extract_features.py $F/sources.jsonl $F/features.jsonl

# the pilot, with timings, both beat devices, and the rest-stem key comparison
uv run --no-project --python 3.12 bench/extract_features.py $F/sources.jsonl $F/pilot.jsonl \
  --keys $F/pilot-keys.txt --device both --timings --compare-rest --threads 2 --batch 5

# spot-check tables for any output file
uv run --no-project --python 3.12 bench/extract_features.py report $F/sources.jsonl $F/pilot.jsonl
```

Options: `--keys FILE` (only these song keys, one per line), `--limit N`, `--batch N` (songs per rsync, default 8), `--threads N` (default 2: torch, numpy and ffmpeg), `--device mps|cpu|both` (default `mps`, with CPU taking over for the rest of the run if mps ever fails), `--src` (rsync source, default `audio-box:./`; a local directory works too).

Resume: songs already in the output are skipped, a half-written last line is cut off, songs that failed (written as `{"key", "error"}`) are retried, and the temp dir is removed on ctrl-C or SIGTERM. Each line is flushed and fsynced. Tested: killed mid-run, resumed, no duplicates.

One run-time trap: a long-lived connection to the box sometimes stalls at ~12 KB/s without failing, while a new one takes seconds. Each rsync therefore runs with `--timeout=15` and retries on a fresh connection.

## Output: one line per song (about 5 KB, so ~28 MB for the library)

```jsonc
{"key": "USQE92000001", "dur": 218.5,
 "tempo": {
   "bpm": 94.9, "bpm_first": 94.9, "bpm_last": 94.9,   // whole song / beats in the first / last 30 s
   "bpm_fold": 94.9, "bpm_ambig": false,               // bpm doubled/halved into [80,160); true if half or double time is plausible
   "bpb": 4,                                           // beats per bar (median)
   "reg": 0.936,                                       // steadiness: share of 4-beat spans within 3% of the median tempo
   "beats": [0.34, 0.96, ...], "downbeats": [0.34, 2.86, ...]},   // seconds, rounded to 10 ms
 "tonal": {"all": ["B", "major", 0.908], "first": [...], "last": [...],   // [tonic, mode, strength], full mix
           "camelot": "1B"},                                              // "rest": [...] only with --compare-rest
 "loud": {"lufs": -8.0, "lra": 7.0,                    // integrated loudness, loudness range
          "st": [-21.5, -20.3, ...],                   // short-term loudness, once per second: st[k] at t = k+1 s
          "typical": -8.4,                             // median second among those within 10 LU of lufs
          "level_t": 11.0,                             // first second from which 3 s stay within 3 LU of typical
          "end": 217.5,                                // where the audio really ends (trailing digital silence cut)
          "tail": {"start": 205.0, "len": 12.5, "drop": 27.2}},   // after the last second within 3 LU of typical; null if < 3 s
 "voc": {"first": 11.8, "last": 205.8, "share": 0.86, "energy": 0.417},
 "t": {"decode": 0.57, ...}}                           // seconds per stage, only with --timings
```

- **Tempo.** beat_this 1.1.0, checkpoint `final0`, `dbn=False`, on the full mix. Its beats sit on a 20 ms grid, so a single interval is off by up to 4% at 120 BPM. BPM is therefore 60 / the median interval over 8-beat spans (error divided by 8), not over single beats. A half/double flag is a rule, not a detection: `bpm_fold` is the tempo moved into [80,160) and `bpm_ambig` is true when that moved it or it sits within 5 BPM of an edge. `bpb` of 3 or 6 flags waltz or compound time.
- **Key.** Essentia `KeyExtractor`, default settings (`bgate` profile), on the mono mix at 44.1 kHz. First/last use the first and last 30 s. `camelot` is the wheel code (relative major/minor share a number).
- **Loudness.** ffmpeg 9.0.1 `ebur128` on the stereo decode (piped, so the file is decoded once). `lufs` and `lra` are ebur128's own summary. ebur128 leaves short-term loudness undefined for the first 3 s, so `st[0]` and `st[1]` use the audio elapsed so far. `level_t`, `end` and `tail` use per-second levels from the 400 ms momentary values and 3 s forward windows: unlike the standard short-term value they do not lag, so times are honest to about a second. `level_t` near 0-1 is a cold start, later a build. A tail with a long `len` and a large `drop` is a fade-out; a long `len` with a small `drop` is a soft outro. There is no fade/quiet label because I could not validate one.
- **Vocals.** The vocals stem decoded to 16 kHz mono, RMS in 50 ms frames. A frame has vocals when the stem is within 20 dB of the full mix in that frame and above -42 dBFS (off below 26 dB under the mix or -48 dBFS). Relative to the mix, because demucs leaves instrument bleed in the stem that fixed dBFS thresholds mistook for singing in quiet songs. Pauses under 1.5 s are bridged, segments under 1 s dropped, and a segment under 3 s with no other vocals within 10 s of it is dropped as a stray chop. `share` is vocal seconds / duration. `energy` is the stem's share of the mix's energy; under 0.02 the song counts as instrumental (`first`/`last` null, `share` 0). The thresholds were tuned on this pilot's 20 songs, so they are not independently validated.

## Pilot

20 songs from 9 playlists (`Everything`, the lofi playlist, `Recent Loop`, two `Top Tracks`, the Slayyyter album, `My 2023 Playlist in a Bottle`, `Japanese`, `MoriDice`): 3 over 5 minutes (Carpenter Brut, Synestia, Balu Brigada; 5:37 to 6:23), 4 instrumentals (Carpenter Brut and three lofi tracks), 4 Spanish, 3 Japanese, the rest English, 2:10 to 6:23. Output: `~/.cache/audio-embed-engine/features/pilot.jsonl`. All 20 succeeded.

```
20 songs

TEMPO                                dur    bpm  first   last   fold amb bpb   reg first~last
Balu Brigada - Backseat              383  125.7  126.3  102.1  125.7       4 0.765     differ
Carpenter Brut - Hang'em All         340  117.1  111.6  142.4  117.1       4 0.568     differ
Mori Calliope - six feet under       217  133.3  133.3  133.3  133.3       4 0.923       same
Mori Calliope - The Grim Reaper is   227  104.8  104.8  105.3  104.8       4 0.997       same
Wakana - Flag                        277  186.0  186.0  186.0   93.0   y   6 0.998       same
Spective - Love You So               172   90.9   90.9   90.9   90.9       4   1.0       same
Bad Bunny, JHAYC - DÁKITI            205  110.1  121.2  110.1  110.1       4  0.86     differ
Eladio Carrion - TQMQA               170  113.2  113.2  113.2  113.2       4 0.866       same
Synestia, Disemb - The Poetic Edda   365  164.4  164.4  128.3   82.2   y   3 0.276     differ
KANKAN, Hiiragi  - Viper             160  167.8  187.5  187.5   83.9   y   4 0.003       same
CMJ, Ben Jammin' - Changing Season   130   81.1   81.1   81.1   81.1   y   4   1.0       same
Klode Chill - Hands Up               135   80.0   80.0   80.0   80.0   y   4 0.988       same
Rauw Alejandro,  - Macacoa 2000 (f   174  100.0  107.1  100.0  100.0       4 0.628     differ
Porter Robinson - Get Your Wish      218   94.9   94.9   94.9   94.9       4 0.936       same
Porter Robinson - Cheerleader        238  148.1  148.1  148.1  148.1       4 0.883       same
Slayyyter - OLD TECHNOLOGY           159  125.0  125.0  125.0  125.0       4 0.985       same
Slayyyter - OLD FLING$               217  118.8  118.8  118.8  118.8       4 0.998       same
Slayyyter - WHAT IS IT LIKE, TO BE   138  115.9  115.9  115.9  115.9       4   1.0       same
STARSET - Unbecoming                 250  129.7  137.9   92.0  129.7       3   0.0     differ
JHAYCO, Bad Bunn - CÓMO SE SIENTE    228   93.0   93.0   93.0   93.0       4   1.0       same

KEY                                       mix  str     first      last cam       rest  mix~rest
Balu Brigada - Backseat                 F min 0.747    Bb min     C min  4A      F min     exact
Carpenter Brut - Hang'em All            E min 0.868     E min     B maj  9A      E min     exact
Mori Calliope - six feet under          C min 0.839     C min     F maj  5A      F maj     other
Mori Calliope - The Grim Reaper is      D min 0.964     D min     D min  7A      D min     exact
Wakana - Flag                           G min 0.924     G min     G min  6A      G min     exact
Spective - Love You So                  A maj 0.824     A maj     A maj 11B      A maj     exact
Bad Bunny, JHAYC - DÁKITI               A min 0.873     A min     C maj  8A      A min     exact
Eladio Carrion - TQMQA                  F maj 0.887     F maj     F maj  7B      F maj     exact
Synestia, Disemb - The Poetic Edda      F maj 0.817     D min    Bb min  7B      F maj     exact
KANKAN, Hiiragi  - Viper               F# min 0.598     C min    F# maj 11A     F# min     exact
CMJ, Ben Jammin' - Changing Season     F# min 0.829    F# min     A maj 11A     F# min     exact
Klode Chill - Hands Up                  E min 0.881     E min     E min  9A      E min     exact
Rauw Alejandro,  - Macacoa 2000 (f      F min 0.867     C min    Bb min  4A      F min     exact
Porter Robinson - Get Your Wish         B maj 0.908     B maj     B maj  1B      B maj     exact
Porter Robinson - Cheerleader          Eb maj 0.934    Eb maj    Eb maj  5B     Eb maj     exact
Slayyyter - OLD TECHNOLOGY             F# min 0.844    F# min     B min 11A     F# min     exact
Slayyyter - OLD FLING$                  A min 0.837     A min     D min  8A      D min     fifth
Slayyyter - WHAT IS IT LIKE, TO BE      A min 0.883     A min     A min  8A      C maj  relative
STARSET - Unbecoming                   Db min 0.959    Db min    Db min 12A     Db min     exact
JHAYCO, Bad Bunn - CÓMO SE SIENTE      Ab min 0.863    Ab min    Eb min  1A     Eb min     fifth

LOUDNESS                            LUFS  LRA level_t    end  tail start+len  drop  whole song (LUFS-20..+3)                         last 30 s (typical-40..+3)
Balu Brigada - Backseat             -7.7  4.4    16.0  383.3          374+9s  16.3  ▆▆███▇▇▇▇▇██████▇▇▇█████████▇▇▇▇▇▇▇▇▇█████████▇▆  ██████████████▇██████▇▇▆▆▆▆▆▆▆
Carpenter Brut - Hang'em All        -4.7  6.4    15.0  335.6         310+26s  44.4  ▅▅██████▆▆▇▇██████▇▆▇▇▇█████████████████████▇▇▆▄  ██▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▅▂▁▁▁▁▁
Mori Calliope - six feet under      -5.5  3.9     3.0  214.8               -        ▆▇▇▇▇▇▇▇▇▇▇████████▇▇▇▇█████▇███████████▇▇█████▇  ██████████████████████████▇▆▅▁
Mori Calliope - The Grim Reaper is -11.6  6.4    18.0  225.6               -        ▃▅▅▅█████▆▇▇▇▇▇▇█▇▇▇█████████▇▇▇▇▇██████████▇▇█▆  █████████████▇▇████████▇█▇▇▇▅▅
Wakana - Flag                       -7.1  6.6     1.0  273.0          270+3s  43.7  ▇███▇▆▆▆▇████████████▆▆▇▇████████████▅▇████████▆  ████████████████████████▇▆▂▁▁▁
Spective - Love You So             -13.8  1.1     0.0  170.4               -        ▇█▇████████▇███████████▇▇█▇████▇▇▇█████████▇███▇  ████████████████████████████▇▁
Bad Bunny, JHAYC - DÁKITI           -8.9  6.8    15.0  204.4         192+12s  35.6  ▆▆▆▆▇▇▇▇█████▇▇▇█████████▇▇▇▆▆▆▆▆████████████▇▃▁  ██████████████████▇▆▆▆▆▅▅▅▄▄▃▂
Eladio Carrion - TQMQA              -8.0  9.8    15.0  169.0          162+7s  32.0  ▃▄▃▄▅███████████▇▇▇▇████▇▇▇████▇██▇▇▇▇▇█████▇▇▆▅  ███████████████▇▇▇██▇▇▇▇▇▆▆▆▅▄
Synestia, Disemb - The Poetic Edda  -6.2  5.7    10.0  363.2         349+14s  49.6  ▂▇▇█████▅▇▇▇▇▇████████▇▇▇▇██████▇███████▆█████▇▁  ███████████████▇▆▅▄▄▄▄▄▃▃▂▂▁▁▁
KANKAN, Hiiragi  - Viper            -7.5  3.7     0.0  156.6               -        ▇███▇▇▇▇▇▇█▇▇████████▇▇▇▇▇▇▇▇▇▇▇▇████████████▆▆▆  ████████████████████▇▇▇▇▇▇▇▇▄▁
CMJ, Ben Jammin' - Changing Season -12.0  2.9     2.0  130.0          123+7s  32.6  ▄▇▇▇▇▇▇█▇█████████████████▇▇▇▇▇▇▇█▇█████████▇▇▆▄  ██████████████████████▇▇▇▇▇▆▅▅
Klode Chill - Hands Up             -10.6  4.3     5.0  133.9               -        ▆▆▆▇▇▆▇▇▆████████▇████████▇████████▇██▇█████████  ████████████████████████████▇▆
Rauw Alejandro,  - Macacoa 2000 (f  -6.9  5.1    10.0  172.3          169+3s  44.6  ▆▆▅▇█████▇████▇▆▆▆████████████████▇▇▇▇▇██████▇▇▆  ██████████████████████████▇▆▅▁
Porter Robinson - Get Your Wish     -8.0  7.0    11.0  217.5         205+12s  27.2  ▄▅▆▇▇▇▇▇▇▇▇▇▇▇▇▇█████████▇▆▇▇▇▇▇▇██▇██████████▆▆  ██████████████████▇▇▇▇▇▇▇▇▇▇▆▅
Porter Robinson - Cheerleader       -5.3 11.7     2.0  236.9               -        ▆███▇▇▇▇▇▇▇██████▆▇▇▇▇▇▇██████▅▅▅▄▄▄▄▇██████████  ██████████████████████████████
Slayyyter - OLD TECHNOLOGY          -7.7  4.9     1.0  157.7          154+4s   5.6  ▆▆▇▇▇▇▇▇▇▇▇▇████▇▇▇▇▇▇▆▇▆▇█████▇███████████████▇  ███████████████████████████▇▆▆
Slayyyter - OLD FLING$              -5.0  4.4     4.0  214.1          210+4s  47.9  ▆▇▆▇▇▇▇▇▇▇▇████▇▇▇▇▇▇████▇▇▇████████▇██████████▂  ████████████████████████▆▅▁▁▁▁
Slayyyter - WHAT IS IT LIKE, TO BE  -7.2  2.5     1.0  136.3               -        ▅▇▇▇▇▇▇██▇██▇██▇██▇████▇▇█▇▇█▇████████████████▇▅  ██████████████████████████▇▇▆▂
STARSET - Unbecoming                -8.7  6.7    28.0  249.2          246+3s  40.9  ▆▆▅▆▆▇▇▇▇█████████▇▇▇▇▇▇██████████▇▆▇▇▇█████████  ██████████████████████████▇▆▂▁
JHAYCO, Bad Bunn - CÓMO SE SIENTE  -10.9  7.6    39.0  227.5         213+14s  18.7  ▅▄▆▆▆▆▆▆▇████████████████████████████▇█▇▆▆▇▇▆▇▆▅  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆

VOCALS                              first   last share energy
Balu Brigada - Backseat              32.8  260.9 0.487  0.151
Carpenter Brut - Hang'em All            -      -   0.0   0.01
Mori Calliope - six feet under        9.2  212.5   0.9  0.391
Mori Calliope - The Grim Reaper is   19.0  223.9 0.902  0.843
Wakana - Flag                         1.9  269.9  0.96  0.628
Spective - Love You So                  -      -   0.0  0.004
Bad Bunny, JHAYC - DÁKITI            16.2  184.7 0.788  0.332
Eladio Carrion - TQMQA               16.3  162.4  0.86  0.717
Synestia, Disemb - The Poetic Edda   12.4  346.6 0.662  0.199
KANKAN, Hiiragi  - Viper             10.4  155.2  0.89   0.41
CMJ, Ben Jammin' - Changing Season      -      -   0.0    0.0
Klode Chill - Hands Up                  -      -   0.0  0.001
Rauw Alejandro,  - Macacoa 2000 (f    2.1  170.1 0.909   0.34
Porter Robinson - Get Your Wish      11.8  205.8  0.86  0.417
Porter Robinson - Cheerleader         0.8  234.3 0.976  0.407
Slayyyter - OLD TECHNOLOGY            5.6  153.6  0.93  0.221
Slayyyter - OLD FLING$                4.5  211.9 0.955  0.411
Slayyyter - WHAT IS IT LIKE, TO BE    1.5  135.6 0.969  0.289
STARSET - Unbecoming                  0.5  240.1 0.847   0.41
JHAYCO, Bad Bunn - CÓMO SE SIENTE     6.6  221.8 0.946  0.272

tempo first vs last 30 s: differ 6, same 14
key mix vs rest (20 songs): {'exact': 16, 'relative': 1, 'fifth': 2, 'parallel': 0, 'other': 1}; exact 80%, exact+relative+fifth 95%

per-song seconds per stage (mean, max):
  beats_cpu        4.43   7.76
  beats_match      1.00   1.00
  beats_mps        1.00   1.87
  decode           0.55   0.97
  fetch            1.64   1.84
  key              0.10   0.16
  loudness         0.49   0.81
  resample         0.02   0.03
  rest_decode      0.52   0.86
  rest_key         0.09   0.15
  vocals           0.02   0.03
  vocals_decode    0.48   0.76
```

`fold`: tempo moved into [80,160). `amb`: half or double time plausible. `first~last`: do the first and last 30 s agree (same = within 2%, octave = double/half). Loudness curves are one character per ~5 s of the whole song, then one per second for the last 30 s.

### Quality checks

- **Tempo, first vs last 30 s.** 14 of 20 agree within 2%. The 6 that do not are the least steady songs: Unbecoming (reg 0.0, beat intervals of 0.44, 0.22 and 0.86 s in different sections, so the tracker flips between octaves), The Poetic Edda (reg 0.28, bpb 3), Hang'em All (0.57), Macacoa 2000 (0.63: 107 then 100), Backseat (0.77), DÁKITI (0.86: 121 then 110). So `reg` does flag them, but I cannot say whether a given disagreement is a tempo change in the song or a tracker error without listening. Flag (186 BPM, fold 93) and Viper (168, fold 84) are marked ambiguous, as half-time is plausible. Get Your Wish comes out at 94.9, which matches my memory of ~95 BPM (not checked against a source).
- **Key, full mix vs `rest` stem** (the mix minus vocals; pilot only): 16 of 20 exact (80%); 1 relative (WHAT IS IT LIKE: A minor vs C major), 2 fifth apart (OLD FLING$ and CÓMO SE SIENTE); 1 other (six feet under: C minor vs F major). Counting relative and fifth as near-misses: 19 of 20 (95%). That checks the two estimates agree, not that either is right. The 30 s windows disagree with the whole song more often (Viper's first 30 s say C minor, the song F# minor), so use `first`/`last` as weak evidence.
- **Vocals on the instrumentals.** All four come out null, share 0 (energy 0.000, 0.001, 0.004, 0.010; every song with vocals is at 0.15 or more). Before the mix-relative rule, fixed -42 dBFS thresholds reported 2 to 27% vocal share on three of them.
- **Vocals vs synced lyrics** (LRCLIB timestamps in `lyrics.jsonl`, 16 songs; the lyrics may be for another edit, and say nothing about wordless vocals). `first` is within 1.5 s of the first lyric line in 8 of 16: DÁKITI, TQMQA, Grim Reaper, Viper (all exact to 0.1 s), Unbecoming, OLD FLING$, Backseat, WHAT IS IT LIKE. Elsewhere the stem finds vocals 3.6 to 22 s before the first lyric line (Macacoa 2000, Cheerleader, Get Your Wish, Poetic Edda, Flag, CÓMO SE SIENTE) or, twice, 2 s after it. I assume intro ad-libs and chops but did not listen. `last` falls within 11 s after (or 1.5 s before) the start of the last lyric line in 13 of 16 (a line takes seconds to sing). The outliers are Backseat (vocals through 260.9 s, lyrics end at 216.9 s, probably vocal chops in its long outro), Cheerleader (+24 s) and Flag (+20 s).
- **Fade-outs vs the loudness curve.** The `tail` start and length line up with the drop in the last-30-s curve for the songs I looked at: Hang'em All drops 44 LU over its last 26 s, DÁKITI and Poetic Edda decay over 12 to 14 s, Flag and Macacoa 2000 end with a 3 s fade. Songs that end on a hit (six feet under, Cheerleader, Viper, WHAT IS IT LIKE) get no tail. Backseat (-16 LU for 9 s) and CÓMO SE SIENTE (-19 LU for 14 s) are soft outros, not fades. What I did not do: confirm any of this by ear, or check the tail rule on a song with a long silent gap in the middle.
- **mps vs CPU beats.** On all 20 songs the beats agree: every beat found by one is found by the other within 25 ms. mps needed no workaround.

## Timings

Pilot, 20 songs (mean 3:40), `--threads 2` on an M1 Pro (8 cores) that other agents were using (load average 4 to 10), so these are pessimistic. Seconds per song, mean (max):

| Stage | mean | max |
| --- | --- | --- |
| fetch from the box (mix + vocals + rest stem, ~8.8 MB, in a prefetch thread) | 1.64 | 1.84 |
| decode mix (ffmpeg, stereo 44.1 kHz) | 0.55 | 0.97 |
| resample to 22.05 kHz | 0.02 | 0.03 |
| beats on **mps** | **1.00** | 1.87 |
| beats on **CPU** (2 threads) | **4.43** | 7.76 |
| key (whole song + first 30 s + last 30 s) | 0.10 | 0.16 |
| loudness (ebur128) | 0.49 | 0.81 |
| decode vocals stem | 0.48 | 0.76 |
| vocals | 0.02 | 0.03 |
| pilot only: decode `rest` stem + key | 0.52 + 0.09 | 0.86 + 0.15 |

mps is 4.4 times faster than CPU on 2 threads. Production mode (mps only, no `rest`, no CPU beats) over the same 20 songs from the box: 67 s wall including start-up and the first fetch, 3.2 s per song overall and about 3.0 steady, with 2.64 s of compute per song. Results were identical to the pilot run. The fetch (~6.2 MB per song without the `rest` stem, ~1.2 s) hides behind compute.

## Estimate for all 5,500 songs

At ~3.0 s per song with 2 threads on a shared machine: about **4.6 hours** (5,500 songs, 34 GB streamed from the box, ~28 MB of output). On an idle Mac with more threads it should be lower, but I did not measure that.

Bottleneck: audio decoding, not the models. The three ffmpeg passes (mix decode, ebur128, vocals decode) are about 1.5 s of the 2.64 s of compute and run one after another; beat tracking on mps is about 1.0 s; key, resampling and vocals are 0.14 s together. The box link is the second limit: ~1.2 s per song, so a slower link (or a stall that retries) would make fetching the bottleneck.

Is it reasonable on the Mac? Yes: a few hours in the background, resumable, 2 threads. Moving it to the Windows PC (RTX 4060) would not pay off: CUDA would cut only the ~1.0 s beat stage (and I did not measure that), leaving ~2 s of ffmpeg decoding, so at best 1.4 times faster. Essentia also ships wheels for macOS and Linux only (checked on PyPI), so the key stage would need WSL or a different key detector. If it needs to go faster, the cheap lever is running two processes on disjoint `--keys` files with separate outputs on the Mac.

## Not verified

- No listening. Every quality statement above is agreement between estimators, or with LRCLIB timestamps; no tempo, key, vocal time or fade was checked against the actual songs or a reference.
- The vocal thresholds and the isolated-segment rule were tuned on these 20 songs; the other 5,480 may need more tuning, particularly songs with backing vocals, choirs or heavy vocal chops.
- Key accuracy on its own: only the agreement between mix and `rest` was measured.
- Timings are for a busy machine at 2 threads. The full run was not attempted, nor the idle-machine or multi-process speed, nor beats on CUDA.
- Songs with tempo changes or rubato (6 of 20 here) can give a `bpm` that describes neither section; use `reg`, `bpm_first` and `bpm_last` together.
- The `final0` checkpoint (81 MB) downloads from `cloud.cp.jku.at` to `~/.cache/torch/hub/checkpoints` on first use. Only the model is fetched; no audio leaves the Mac.
