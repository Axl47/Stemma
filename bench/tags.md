# Mood, genre and voice tags from the stored vectors

`bench/tags.py` runs Essentia's pretrained classification heads on the per-patch Discogs-EffNet embeddings already stored in `~/.cache/audio-embed-engine/patches/` (one 1280-d vector per ~2 s of each full track, written by `embed_patches.py`) and averages each head's output over the song's patches, the Essentia convention. No audio is decoded, so the whole library takes about 15 s. Head files come from essentia.upf.edu into `~/.cache/audio-embed-engine/tags/models/`; no library data leaves the machine.

## Run

```sh
cd /Users/axel/Desktop/Code_Projects/Personal/Developer/Stemma
uv run --no-project --python 3.12 bench/tags.py run        # writes ~/.cache/audio-embed-engine/tags/tags.jsonl from scratch
uv run --no-project --python 3.12 bench/tags.py verify     # recompute 8 songs from audio, compare (~30 s, needs audio-box)
uv run --no-project --python 3.12 bench/tags.py report /Users/axel/Desktop/Code_Projects/Personal/Web/Cento/data
```

The script carries its own dependencies (PEP 723: essentia, onnx, onnxruntime, numpy, soxr), so it needs neither the project's virtualenv nor `uv sync`. `run` always rewrites the file (via `tags.jsonl.part`); rerun it after new patches are embedded.

## Heads

All are the `-discogs-effnet-1` variants, run with onnxruntime on CPU. Binary heads store the probability of the class in the key.

| key | head | note |
| --- | --- | --- |
| `genre_discogs400` | EffNet's own 400-style classifier | The separate `genre_discogs400-discogs-effnet-1` head is published only as a TensorFlow `.pb`. EffNet's `activations` output is `sigmoid(embeddings @ W.T + b)` (a single Gemm in the ONNX graph), so the script reads `W`, `b` from the EffNet ONNX and applies them to stored embeddings. Same 400 classes; this is also what the engine's `--tags` uses. |
| `mtg_jamendo_moodtheme` | mtg_jamendo_moodtheme | 56 sigmoid outputs |
| `happy` `sad` `relaxed` `aggressive` `party` `acoustic` `electronic` | mood_* | |
| `danceable` | danceability | |
| `approachable` `engaging` | approachability_2c, engagement_2c | the 2-class variants, for one probability each |
| `voice` | voice_instrumental | |
| `female` | gender | voice gender; meaningless on instrumentals |
| `bright` | timbre | |
| `tonal` | tonal_atonal | |

Arousal/valence heads (deam, emomusic, muse) exist only for the audioset-vggish and msd-musicnn backbones, so they are skipped.

## Output: `~/.cache/audio-embed-engine/tags/tags.jsonl`

One line per song (about 1.6 KB; 9 MB for the library):

```jsonc
{"key": "USQX92605161",          // engine key: uppercase ISRC, or the Spotify id when there is none
 "n": 97,                        // patches averaged; 0 for preview-only songs
 // "preview": true,             // only on preview-only songs, see below
 "genre_discogs400": [["Hip Hop---Trap", 0.198], ["Hip Hop---Cloud Rap", 0.155], ...],  // top 8 of 400, descending
 "mtg_jamendo_moodtheme": {"action": 0.015, ..., "uplifting": 0.01},                  // all 56
 "happy": 0.243, "sad": 0.05, "relaxed": 0.218, "aggressive": 0.805, "party": 0.773, "acoustic": 0.014,
 "electronic": 0.936, "danceable": 0.965, "approachable": 0.228, "engaging": 0.939, "voice": 0.804,
 "female": 0.408, "bright": 0.449, "tonal": 0.002}     // mean over patches, rounded to 3 decimals
```

Coverage on 2026-11-12: 5,481 full tracks plus 365 preview-only songs (5,846 lines).

**Preview-only songs** have no patches, only the engine's 30 s preview vector, which is the L2-normalised mean embedding. It is rescaled to the median norm of full-track mean embeddings (3.27) and run through the heads as a single row. That is an approximation twice over: a head of the mean is not the mean of the head, and 30 s is not the whole song. Measured on full tracks in `verify`, the single-row shortcut alone (before the 30 s cut) moves binary heads by 0.01 to 0.14 on average (worst song 0.29), styles and moods by 0.004 and 0.017. Treat `"preview": true` lines as rough.

## Are the stored patches valid head inputs? (`verify`)

`embed_patches.py` stores the model's raw `embeddings` output (not normalised; row norms around 5), non-overlapping 128-frame patches (hop 128), from the engine's decode (PyAV, channels averaged, soxr HQ to 16 kHz) and the numpy mel that matches Essentia's `TensorflowInputMusiCNN`. `verify` picks 8 random songs, fetches their audio from the box into a temp dir, and checks two things:

- Recomputing with the engine's decode and hop 128 reproduces the stored rows exactly (cosine 1.0000, norm ratio 1.0000 on all 8). A trap found on the way: ffmpeg's `-ac 1` mixes stereo as (L+R)/√2, 3 dB louder than averaging, and that alone moved embeddings to cosine 0.94 and norms by up to 14%. EffNet is level-sensitive.
- Against the Essentia reference path (`MonoLoader` at 16 kHz, patch hop 62 as in `TensorflowPredictEffnetDiscogs`), the song-level head outputs from stored patches differ by a mean absolute 0.002 to 0.028 per binary head (worst single song 0.071, `voice`), 0.001 for moods and styles (worst single class 0.048), and the top style matched on 8 of 8 songs.

So the stored patches are valid inputs; the remaining difference is overlap (hop 62 vs 128) and resampler, well below what any of these heads can resolve.

## Do the tags mean anything on this library? (`report`)

Measured on the 40 largest playlists plus 13 smaller ones with genre or activity names (playlist mean / library mean):

- **Styles separate genres sharply.** Urban playlists (Bad Bunny, Mora, Urbano, Regueton): Reggaeton 0.4 to 0.6 and Trap 0.3 to 0.46 against 0.09 and 0.11. Rock, Flow, Gym, Lock In, PTV/BMTH: Metalcore, Post-Hardcore, Alternative Rock, Nu Metal. Japanese and J-Rock: J-pop 0.22 and 0.16 against 0.02. Lo-fi: Instrumental, Jazzy Hip-Hop, Ambient, Chillwave. Chill Breakcore: Drum n Bass, Jungle, Breakcore. Dance: House, Electro. Spot checks are right for Linkin Park (Nu Metal), BMTH (Metalcore), YOASOBI (J-pop), Sophie Ellis-Bextor (House, Disco), Arcángel and Feid (Reggaeton).
- **Styles fail in known places.** There is no Future Bass class, so that playlist reads Dubstep/Electro House. K-pop and Cloud Rap act as catch-alls: Tame Impala's "The Less I Know The Better" is K-pop/Electro, Clairo's "Flaming Hot Cheetos" is Juke, and K-pop tops "Happy :)" and "Soft Female Vibes".
- **Strong mood heads:** `aggressive` (rock playlists 0.6 to 0.87, urban 0.05 to 0.1, library 0.22), `relaxed` (Lo-fi 0.95, Chaos 0.95, Chill 0.83, Rock 0.28), `sad` (Sad 0.56, Chaos 0.79, EDEN 0.51, against 0.28), `acoustic`, `party` and `voice` (lo-fi 0.10 to 0.16, Chill Breakcore 0.34, Hamilton 0.95).
- **Weak:** `happy`. The owner's "Happy :)" playlist scores 0.34 against a library mean of 0.33; it rises only on Regueton Antiguo (0.64), J-Rock (0.60) and Sophie Ellis-Bextor (0.81). `danceable` is saturated (library mean 0.81) but drops on Sad (0.56) and Chaos (0.30).
- **Noise or miscalibrated:** `bright` (library std 0.03: effectively constant). `tonal` (library mean 0.07, i.e. nearly all pop called atonal; its variation tracks acoustic/orchestral material, Hamilton 0.30, Chaos 0.22, so it is not usable as labelled).
- **`female`** works on the extremes (Soft Female Vibes 0.78, male urban and rock 0.27 to 0.30) but misses individual songs (Tame Impala 0.70, XXXTENTACION "Looking for a Star" 0.68).
- **Jamendo moods** have small absolute values (top tags about 0.1 to 0.2) and "energetic" and "love" lead almost every playlist, so read them as lift over the library mean: heavy ×4.8 to ×7 on rock, calm ×3 to ×4 on lo-fi, game and retro ×4.6 and ×3.3 on Sonic, trailer and film ×6 on Chaos, ballad ×3.7 on Hamilton. "christmas" and "holiday" lift on any soft or orchestral playlist and should be ignored.
