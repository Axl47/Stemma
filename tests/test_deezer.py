import httpx

from audio_embed_engine.deezer import Deezer, Preview
from audio_embed_engine.types import Track

TRACK = Track(isrc="USX", title="Song - Remastered 2011", artists=("The Band",), duration_ms=200_000)


def client(handler) -> Deezer:
    return Deezer(httpx.Client(transport=httpx.MockTransport(handler)), window=0.01)


def test_quota_error_is_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if len(calls) == 1:
            return httpx.Response(200, json={"error": {"code": 4, "message": "Quota limit exceeded"}})
        return httpx.Response(200, json={"id": 7, "preview": "https://cdn/p.mp3"})

    assert client(handler).resolve(TRACK) == Preview(7, "https://cdn/p.mp3")
    assert calls == ["/track/isrc:USX", "/track/isrc:USX"]


def test_search_fallback_matches_duration_and_artist() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/track/isrc:"):
            return httpx.Response(200, json={"error": {"code": 800, "message": "no data"}})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": 1,
                        "duration": 260,
                        "artist": {"name": "The Band"},
                        "preview": "https://cdn/1.mp3",
                    },  # wrong length
                    {
                        "id": 2,
                        "duration": 202,
                        "artist": {"name": "Someone Else"},
                        "preview": "https://cdn/2.mp3",
                    },  # wrong artist
                    {"id": 3, "duration": 198, "artist": {"name": "The Band"}, "preview": "https://cdn/3.mp3"},
                ]
            },
        )

    assert client(handler).resolve(TRACK) == Preview(3, "https://cdn/3.mp3")


def test_found_without_preview_is_no_preview() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/track/isrc:"):
            return httpx.Response(200, json={"id": 1, "preview": ""})
        return httpx.Response(200, json={"data": []})

    assert client(handler).resolve(TRACK) == "no_preview"
