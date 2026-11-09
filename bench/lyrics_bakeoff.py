"""Compare lyric transcription models on songs that already have LRCLIB lyrics.

Usage: python bench/lyrics_bakeoff.py transcribe <engine> <mix|vocals> <sample.jsonl> <mix_dir> <stems_dir> <vectors_dir> <out.jsonl>
       python bench/lyrics_bakeoff.py score <sample.jsonl> <result.jsonl> [<result.jsonl> ...]

Engines: whisper:<faster-whisper model>, heart:<checkpoint path>, qwen:<model id>. Every song is cut into sections of at
most 30 s from the loudness of its separated vocals (arXiv 2506.15514: normalised RMS, 0.1 threshold, 1 s minimum
silence), and the language is detected rather than supplied. "transcribe" resumes from existing lines.

"score" reports language accuracy first, then invented text, then errors: word error rate (character error rate for
Japanese), content-word recall, IR10 (inserted runs of 10+ tokens, per reference token), and how much text appears on
instrumental tracks with and without a gate on the song's vocal energy.
"""

import argparse
import json
import re
import statistics as st
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RATE = 16_000
FRAME, HOP = 400, 160
THRESHOLD, MIN_SILENCE_S, MAX_SECTION_S, PAD_S = 0.1, 1.0, 30.0, 0.2
SECTIONS_FOR_LANGUAGE = 5
GATES = (0.005, 0.01, 0.02, 0.05)
NAMES = {"spanish": "es", "english": "en", "japanese": "ja", "korean": "ko", "portuguese": "pt", "french": "fr", "german": "de", "italian": "it"}


def load_mono(path: Path) -> np.ndarray:
    import av
    import soxr

    planar = av.AudioResampler(format="fltp")
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        rate = s.rate
        pcm = np.concatenate([f.to_ndarray() for fr in c.decode(s) for f in planar.resample(fr)], axis=1).mean(0)
    return soxr.resample(pcm, rate, RATE, quality="HQ").astype(np.float32)


def sections(vocals: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges where the vocals are active, merged across short gaps and split at the quietest point past 30 s."""
    if len(vocals) < FRAME:
        return []
    n = (len(vocals) - FRAME) // HOP + 1
    power = np.convolve(vocals**2, np.ones(FRAME) / FRAME, "valid")[::HOP][:n]
    rms = np.sqrt(power) / max(float(np.sqrt(power.max())), 1e-9)
    active = np.flatnonzero(rms > THRESHOLD)
    if not len(active):
        return []
    gap = int(MIN_SILENCE_S * RATE / HOP)
    runs, start, last = [], int(active[0]), int(active[0])
    for f in map(int, active[1:]):
        if f - last > gap:
            runs.append((start, last + 1))
            start = f
        last = f
    runs.append((start, last + 1))

    limit, out = int(MAX_SECTION_S * RATE / HOP), []
    for a, b in runs:
        while b - a > limit:
            low = a + limit // 2
            cut = low + int(np.argmin(rms[low : a + limit]))
            out.append((a, cut))
            a = cut
        out.append((a, b))
    pad = int(PAD_S * RATE)
    return [(max(a * HOP - pad, 0), min(b * HOP + FRAME + pad, len(vocals))) for a, b in out]


def whisper_engine(name: str):
    from faster_whisper import WhisperModel

    model = WhisperModel(name, device="cuda", compute_type="int8_float16")
    options = {"beam_size": 5, "temperature": 0.0, "condition_on_previous_text": False, "vad_filter": False}

    def language(chunks: list[np.ndarray]) -> list[str]:
        return [model.transcribe(c, **options)[1].language for c in chunks]

    def transcribe(chunks: list[np.ndarray], lang: str) -> list[str]:
        out = []
        for c in chunks:
            segments, _ = model.transcribe(c, language=lang, **options)
            # at temperature 0 Whisper has no retry, so its own failure signals have to be applied here
            out += [s.text for s in segments if s.compression_ratio <= 2.4 and not (s.no_speech_prob > 0.6 and s.avg_logprob < -1)]
        return out

    return language, transcribe


def heart_engine(path: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()

    def features(chunk: np.ndarray) -> torch.Tensor:
        return processor(chunk, sampling_rate=RATE, return_tensors="pt").input_features.cuda().half()

    def language(chunks: list[np.ndarray]) -> list[str]:
        out = []
        for c in chunks:
            with torch.no_grad():
                token = model.detect_language(features(c))[0]
            out.append(processor.tokenizer.decode(token).strip("<|>"))
        return out

    def transcribe(chunks: list[np.ndarray], lang: str) -> list[str]:
        out = []
        for c in chunks:
            with torch.no_grad():
                ids = model.generate(features(c), task="transcribe", language=lang, num_beams=5, temperature=0.0, do_sample=False)
            out.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        return out

    return language, transcribe


def qwen_engine(name: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(name, dtype=torch.float16, device_map="cuda:0", max_inference_batch_size=1, max_new_tokens=256)
    cache: dict[int, object] = {}

    def run(c: np.ndarray):
        return cache.setdefault(id(c), model.transcribe(audio=[(c, RATE)], language=None)[0])

    def language(chunks: list[np.ndarray]) -> list[str]:
        return [NAMES.get(str(run(c).language).lower(), str(run(c).language).lower()) for c in chunks]

    def transcribe(chunks: list[np.ndarray], lang: str) -> list[str]:
        out = [run(c).text for c in chunks]
        cache.clear()
        return out

    return language, transcribe


ENGINES = {"whisper": whisper_engine, "heart": heart_engine, "qwen": qwen_engine}


def cmd_transcribe(a: argparse.Namespace) -> None:
    kind, name = a.engine.split(":", 1)
    done = {json.loads(line)["key"] for line in open(a.out, encoding="utf-8")} if a.out.exists() else set()
    sample = [json.loads(line) for line in open(a.sample, encoding="utf-8")]
    language, transcribe = ENGINES[kind](name)
    with open(a.out, "a", encoding="utf-8") as out:
        for i, s in enumerate(sample, 1):
            if s["key"] in done:
                continue
            vocals = load_mono(a.stems / f"{s['key']}.vocals.opus")
            audio = vocals if a.input == "vocals" else load_mono(a.mix / s["file"])
            spans = sections(vocals)
            chunks = [audio[x:y] for x, y in spans]
            start = time.perf_counter()
            longest = sorted(chunks, key=len, reverse=True)[:SECTIONS_FOR_LANGUAGE]
            votes = language(longest) if longest else []
            lang = Counter(votes).most_common(1)[0][0] if votes else None
            text = " ".join(t.strip() for t in transcribe(chunks, lang)) if chunks else ""
            energy = float(np.load(a.vectors / f"{s['key']}.npz")["vocal_energy"])
            row = {"key": s["key"], "language": lang, "votes": votes, "text": text, "seconds": time.perf_counter() - start,
                   "audio_s": round(len(audio) / RATE, 1), "sung_s": round(sum(y - x for x, y in spans) / RATE, 1), "vocal_energy": energy}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{i}/{len(sample)} {s['key']} {lang}", flush=True)


def normalize(text: str) -> str:
    """NFKC, lowercased, section tags and punctuation removed. Accents and ñ are kept."""
    text = re.sub(r"\[[^\]]*\]", " ", unicodedata.normalize("NFKC", text).lower())
    return " ".join("".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text).split())


def tokens(text: str, language: str) -> list[str]:
    n = normalize(text)
    return list(n.replace(" ", "")) if language == "ja" else n.split()


def score_one(ref: list[str], hyp: list[str]) -> dict[str, float]:
    from rapidfuzz.distance import Levenshtein

    inserted = sum(op.dest_end - op.dest_start for op in Levenshtein.opcodes(ref, hyp) if op.tag == "insert" and op.dest_end - op.dest_start >= 10)
    content = {t for t in ref if len(t) >= 4}
    return {"wer": Levenshtein.distance(ref, hyp) / max(len(ref), 1), "ir10": inserted / max(len(ref), 1),
            "recall": len(content & set(hyp)) / len(content) if content else float("nan")}


def cmd_score(a: argparse.Namespace) -> None:
    sample = {s["key"]: s for s in map(json.loads, open(a.sample, encoding="utf-8"))}
    for path in a.results:
        rows = [r for r in map(json.loads, open(path, encoding="utf-8")) if r["key"] in sample]
        by: dict[str, list[dict[str, float]]] = defaultdict(list)
        right, known = Counter(), Counter()
        instrumental = []
        for r in rows:
            s = sample[r["key"]]
            if s["language"] == "instrumental":
                instrumental.append(r)
                continue
            if s["language"] in ("es", "en", "ja"):
                known[s["language"]] += 1
                right[s["language"]] += r["language"] == s["language"]
            by[s["language"]].append(score_one(tokens(s["plain"], s["language"]), tokens(r["text"], s["language"])))
        total_s, audio_s = sum(r["seconds"] for r in rows), sum(r["audio_s"] for r in rows)
        print(f"\n{Path(path).stem}: {len(rows)} songs, {audio_s / max(total_s, 1e-9):.1f}x real time")
        print("  language right: " + ", ".join(f"{k} {right[k]}/{known[k]}" for k in known) + f", all {sum(right.values())}/{sum(known.values())}")
        for lang, m in sorted(by.items()):
            print(f"  {lang:>5}: n={len(m):>3}  median error {st.median(x['wer'] for x in m) * 100:5.1f}%  "
                  f"mean IR10 {st.mean(x['ir10'] for x in m) * 100:5.1f}%  content recall {st.mean(x['recall'] for x in m if x['recall'] == x['recall']) * 100:5.1f}%")
        if instrumental:
            words = [len(r["text"].split()) for r in instrumental]
            print(f"  instrumentals: {sum(w >= 20 for w in words)}/{len(words)} produced 20+ words, median {st.median(words)} words")
            sung = [r for r in rows if sample[r["key"]]["language"] != "instrumental"]
            for g in GATES:
                kept = sum(r["vocal_energy"] < g and len(r["text"].split()) >= 20 for r in instrumental)
                lost = sum(r["vocal_energy"] < g for r in sung)
                print(f"    gate vocal energy >= {g}: instrumentals blocked {sum(r['vocal_energy'] < g for r in instrumental)}/{len(instrumental)} "
                      f"({kept} of them had text), real songs wrongly blocked {lost}/{len(sung)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("engine")
    t.add_argument("input", choices=["mix", "vocals"])
    for name, kind in (("sample", Path), ("mix", Path), ("stems", Path), ("vectors", Path), ("out", Path)):
        t.add_argument(name, type=kind)
    t.set_defaults(func=cmd_transcribe)
    s = sub.add_parser("score")
    s.add_argument("sample", type=Path)
    s.add_argument("results", type=Path, nargs="+")
    s.set_defaults(func=cmd_score)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
