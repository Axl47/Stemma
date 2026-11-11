"""Adds usable Whisper transcripts to the lyrics file, replacing only `not_found` rows.

Usage: python bench/merge_transcripts.py <transcripts.jsonl> <lyrics.jsonl> [<out.jsonl>]  (no out: dry run with stats)
Added rows are {"key", "status": "ok", "via": "whisper-large-v3-vocals", "plain", "language"}.
"""

import json
import re
import sys
from collections import Counter

MIN_VOCAL_ENERGY = 0.05
MIN_WORDS, MIN_CHARS = 20, 80  # spaced scripts count words; Japanese and the like count characters
MIN_UNIQUE_SHARE = 0.3  # "heaps, heaps, heaps" has words but says nothing
GIBBERISH = {"la", "jw"}  # Whisper's labels for invented-language vocals in soundtracks
# Whisper was trained on subtitled video, so it invents these over quiet or wordless audio.
CREDIT = re.compile(
    r"thanks? (you )?for watching|subscribe|субтитры|подпис|редактор|корректор|obrigado|gracias por ver|suscr[ií]bete|اشتركوا|موسيقى|©|"
    r"bf-watch|satsang|ご視聴|チャンネル登録|字幕|amara\.org|시청|구독|www\.|\.com|^música$|^musique$|^music$",
    re.IGNORECASE,
)
FILLER = {"thank you", "thanks", "you", "bye", "bye bye", "all right", "okay", "oh", "oh my god", "gracias", "um", "tchau", "boa noite"}
LINE_WORDS, LINE_CHARS = 8, 24


def clean(text: str) -> str:
    kept = []
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        bare = re.sub(r"[^\w\s]", " ", sentence.lower()).strip()
        bare = " ".join(bare.split())
        if bare and bare not in FILLER and not CREDIT.search(sentence.strip(" .!?。！？")):
            kept.append(sentence)
    return " ".join(kept)


def usable(text: str) -> bool:
    words = text.split()
    spaced = len(words) >= MIN_WORDS
    if not spaced and len(re.sub(r"\s", "", text)) < MIN_CHARS:
        return False
    if not spaced:
        return True
    return len({w.lower().strip(".,!?") for w in words}) / len(words) >= MIN_UNIQUE_SHARE


def as_lines(text: str) -> str:
    """Whisper gives one run of text; the passage splitter expects lines."""
    lines, current = [], []
    for word in text.split():
        pieces = [word[i : i + LINE_CHARS] for i in range(0, len(word), LINE_CHARS)]
        for piece in pieces:
            current.append(piece)
            if len(current) == LINE_WORDS or len(piece) >= LINE_CHARS:
                lines.append(" ".join(current))
                current = []
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


why, added = Counter(), {}
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    if row["vocal_energy"] < MIN_VOCAL_ENERGY:
        why["little vocal energy"] += 1
    elif row["language"] in GIBBERISH:
        why["invented or unclear language"] += 1
    elif not usable(text := clean(row["text"])):
        why["too little real text"] += 1
    else:
        added[row["key"]] = {"key": row["key"], "status": "ok", "via": "whisper-large-v3-vocals", "plain": as_lines(text), "language": row["language"]}
        why["added"] += 1

out = open(sys.argv[3], "w", encoding="utf-8") if len(sys.argv) > 3 else None
replaced = 0
for line in open(sys.argv[2], encoding="utf-8"):
    row = json.loads(line)
    if row["status"] == "not_found" and row["key"] in added:
        replaced += 1
        line = json.dumps(added[row["key"]], ensure_ascii=False) + "\n"
    if out:
        out.write(line)
print(dict(why), "replaced", replaced)
print(Counter(r["language"] for r in added.values()).most_common())
if not out:
    common = Counter(s.strip().lower() for r in added.values() for s in re.split(r"\n", r["plain"]))
    print([x for x in common.most_common(15)])
    sentences = Counter()
    for r in added.values():
        for s in set(re.split(r"(?<=[.!?。！？])\s+", r["plain"].replace("\n", " "))):
            sentences[s.lower()] += 1
    print([x for x in sentences.most_common(30) if x[1] > 2])
