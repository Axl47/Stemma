"""Build a labelled benchmark track list from Deezer (one-off; output is committed).

Picks each artist's top tracks so we get same-artist and same-style pairs across
clearly different styles. Writes engine-format JSONL with an extra `style` field.
"""

import json
import sys

import httpx

STYLES: dict[str, list[str]] = {
    "synthwave": ["The Midnight", "FM-84", "Gunship"],
    "acoustic-folk": ["Iron & Wine", "Fleet Foxes", "Gregory Alan Isakov"],
    "metal": ["Lamb of God", "Mastodon", "Slipknot"],
    "lofi-hiphop": ["Nujabes", "Jinsang", "Tomppabeats"],
    "jazz": ["Miles Davis", "Bill Evans", "John Coltrane"],
    "house": ["Purple Disco Machine", "Fred again..", "Kaytranada"],
    "country": ["Chris Stapleton", "Luke Combs", "Zach Bryan"],
    "reggaeton": ["Bad Bunny", "J Balvin", "Daddy Yankee"],
    "classical-piano": ["Ludovico Einaudi", "Yiruma", "Lang Lang"],
}
PER_ARTIST = 2


def main(out_path: str) -> None:
    rows = []
    with httpx.Client(base_url="https://api.deezer.com", timeout=20) as c:
        for style, artists in STYLES.items():
            for name in artists:
                found = c.get("/search/artist", params={"q": name, "limit": 5}).json()["data"]
                artist = next(a for a in found if a["name"].casefold() == name.casefold())
                top = c.get(f"/artist/{artist['id']}/top", params={"limit": 10}).json()["data"]
                picked = 0
                for t in top:
                    full = c.get(f"/track/{t['id']}").json()
                    if not full.get("isrc") or not full.get("preview"):
                        continue
                    rows.append(
                        {
                            "isrc": full["isrc"],
                            "title": full["title"],
                            "artists": [a["name"] for a in full.get("contributors", [])] or [artist["name"]],
                            "durationMs": full["duration"] * 1000,
                            "style": style,
                            "artistLabel": artist["name"],
                        }
                    )
                    picked += 1
                    if picked == PER_ARTIST:
                        break
                print(style, artist["name"], picked, file=sys.stderr)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


if __name__ == "__main__":
    main(sys.argv[1])
