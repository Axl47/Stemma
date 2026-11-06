"""Decoding and the mel-spectrogram front end shared by the Essentia models.

`log_mel` reimplements Essentia's TensorflowInputMusiCNN (512-sample Hann frames, hop 256,
96 Slaney mel bands over 0-8 kHz, log10(1 + 10000 * power)) in numpy, so the engine
needs no Essentia install. tests/test_audio.py checks it against Essentia itself.
"""

import io

import numpy as np
import soundfile as sf
import soxr

type Audio = np.ndarray  # float32 mono at the model's sample rate
SAMPLE_RATE = 16_000
FRAME, HOP, BANDS = 512, 256, 96
MIN_SECONDS = 3.0  # previews are ~30 s; anything this short is broken (and EffNet needs >= 2.05 s)


class DecodeError(Exception):
    pass


def _strip_id3(data: bytes) -> bytes:
    """Drop a leading ID3v2 tag. Deezer previews carry an empty one that libsndfile can't sniff past in memory."""
    if data[:3] != b"ID3" or len(data) < 10:
        return data
    size = int.from_bytes(bytes(b & 0x7F for b in data[6:10]), "big")  # syncsafe integer
    return data[10 + size :]


def decode(data: bytes, sample_rate: int = SAMPLE_RATE) -> Audio:
    """MP3/any libsndfile format bytes -> mono float32 resampled to `sample_rate`."""
    try:
        with sf.SoundFile(io.BytesIO(_strip_id3(data))) as f:
            pcm = f.read(dtype="float32", always_2d=True)
            source_rate = f.samplerate
    except Exception as e:  # libsndfile raises a grab bag of types on bad input
        raise DecodeError(str(e)) from e
    if len(pcm) < MIN_SECONDS * source_rate:
        raise DecodeError(f"audio too short: {len(pcm) / source_rate:.1f} s")
    return soxr.resample(pcm.mean(axis=1), source_rate, sample_rate, quality="HQ").astype(np.float32)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    # Slaney: linear below 1 kHz, logarithmic above.
    return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27), f / (200 / 3))


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return np.where(m >= 15, 1000 * np.exp((m - 15) * np.log(6.4) / 27), m * 200 / 3)


def _filterbank() -> np.ndarray:
    """(BANDS, FRAME/2+1) triangular filters with unit area ('unit_tri' in Essentia)."""
    edges = _mel_to_hz(np.linspace(0, _hz_to_mel(np.array(SAMPLE_RATE / 2)), BANDS + 2))
    bins = np.arange(FRAME // 2 + 1) * SAMPLE_RATE / FRAME
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    tri = np.maximum(0, np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)))
    return (tri * (2 / (hi - lo))).astype(np.float32)


_FILTERS = _filterbank()
_WINDOW = np.hanning(FRAME).astype(np.float32)


def log_mel(audio: Audio) -> np.ndarray:
    """(frames, 96) log-mel spectrogram; frames are centred on multiples of HOP (zero padded)."""
    padded = np.pad(audio, (FRAME // 2, FRAME // 2))
    n = 1 + (len(padded) - FRAME) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, FRAME)[::HOP][:n] * _WINDOW
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log10(1 + 10000 * (power @ _FILTERS.T)).astype(np.float32)


def patches(mel: np.ndarray, size: int, hop: int) -> np.ndarray:
    """(n, size, bands) overlapping windows over time; trailing frames that don't fill a patch are dropped."""
    if len(mel) < size:
        raise DecodeError(f"audio too short: {len(mel)} frames < {size}")
    return np.lib.stride_tricks.sliding_window_view(mel, (size, mel.shape[1]))[::hop, 0]
