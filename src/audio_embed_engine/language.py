"""Guess which language a song's lyrics are in, the one thing the sound model can't tell the classifier."""

import re

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # the detector samples randomly; a fixed seed gives the same answer every run

MIN_WORDS = 30
MIN_PROBABILITY = 0.7
_SAMPLE_CHARS = 3000
# Statistical detection confuses short CJK text, but these two scripts each belong to one language.
_KANA, _HANGUL = re.compile(r"[぀-ヿ]"), re.compile(r"[가-힯]")


def detect_language(lyrics: str) -> str | None:
    """ISO 639-1 code for the lyrics, or None when they're too short or the detector is unsure."""
    if len(lyrics.split()) < MIN_WORDS:
        return None
    if _KANA.search(lyrics):
        return "ja"
    if _HANGUL.search(lyrics):
        return "ko"
    try:
        best = detect_langs(lyrics[:_SAMPLE_CHARS])[0]
    except LangDetectException:
        return None
    return best.lang if best.prob >= MIN_PROBABILITY else None
