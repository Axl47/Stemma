import io

import numpy as np
import pytest
import soundfile as sf

from audio_embed_engine.audio import DecodeError, decode, log_mel, patches


def _signal(seconds: float = 6.0, rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(0)
    chirp = np.sin(2 * np.pi * (200 + 600 * t) * t)
    return (0.5 * chirp + 0.05 * rng.normal(size=t.size)).astype(np.float32)


def test_log_mel_matches_essentia() -> None:
    es = pytest.importorskip("essentia.standard")
    audio = _signal()
    extractor = es.TensorflowInputMusiCNN()
    ref = np.array([extractor(f) for f in es.FrameGenerator(audio, frameSize=512, hopSize=256)])
    ours = log_mel(audio)
    n = min(len(ref), len(ours))  # Essentia emits one extra trailing frame; patches drop it anyway
    assert abs(len(ref) - len(ours)) <= 1
    np.testing.assert_allclose(ours[:n], ref[:n], atol=1e-3)


def test_decode_mp3_with_empty_id3_header() -> None:
    # Deezer previews start with a zero-length ID3v2 tag, which libsndfile can't sniff in memory.
    buf = io.BytesIO()
    sf.write(buf, np.tile(_signal(3.0, 44_100)[:, None], 2), 44_100, format="MP3")
    data = b"ID3\x04\x00\x00\x00\x00\x00\x00" + buf.getvalue()
    audio = decode(data, 16_000)
    assert audio.dtype == np.float32
    assert abs(len(audio) - 3 * 16_000) < 2_000  # MP3 encoder padding


def test_decode_garbage_raises() -> None:
    with pytest.raises(DecodeError):
        decode(b"not audio at all" * 100)


def test_patches_shape_and_short_audio() -> None:
    mel = np.zeros((300, 96), np.float32)
    assert patches(mel, 128, 128).shape == (2, 128, 96)
    with pytest.raises(DecodeError):
        patches(mel[:100], 128, 128)
