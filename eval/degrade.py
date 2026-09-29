"""Audio degradations that simulate real-world query conditions.

Each function takes mono float32 samples at ``sr`` and returns a degraded copy
of the same length (approximately, for the codec round trip).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


def add_white_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at ``snr_db`` relative to the clip's own power.

    SNR is measured against the clip (not the whole song), so a quiet passage
    gets proportionally quieter noise, and the SNR is the one the matcher sees.
    At 0 dB the noise is as loud as the music; at -5 dB it is ~3x louder.
    """
    p_signal = float(np.mean(np.square(x, dtype=np.float64)))
    if p_signal == 0.0:
        return x.copy()
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    return (x + rng.standard_normal(x.size) * np.sqrt(p_noise)).astype(np.float32)


def band_pass(x: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """4th-order Butterworth band-pass: the classic telephone band.

    A causal filter (``sosfilt``, not the zero-phase ``sosfiltfilt``), because a
    real microphone/speaker chain is causal too. Its small group delay shifts
    peaks by well under one STFT frame.
    """
    sos = butter(4, [low_hz, min(high_hz, sr / 2 * 0.99)], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def low_pass(x: np.ndarray, sr: int, cutoff_hz: float = 3400.0) -> np.ndarray:
    """4th-order Butterworth low-pass (small speakers / cheap mics lose the highs)."""
    sos = butter(4, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def mp3_roundtrip(x: np.ndarray, sr: int, bitrate: str = "32k") -> np.ndarray:
    """Encode to MP3 with LAME and decode back (lossy-codec artifacts).

    Goes through a real file so the decoder can read the LAME header and trim
    the encoder's priming delay; piping would shift the audio by ~0.1 s.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "clip.mp3"
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
             "-i", "-", "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3)],
            input=x.astype("<f4").tobytes(), check=True,
        )
        out = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(mp3), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, check=True,
        ).stdout
    y = np.frombuffer(out, dtype="<f4")
    return y[: x.size].astype(np.float32)


def phone_mic(x: np.ndarray, sr: int, rng: np.random.Generator, snr_db: float = 10.0) -> np.ndarray:
    """Rough "played on one device, recorded on a phone" chain.

    Telephone band-pass, then room/mic noise at ``snr_db``, then a
    low-bitrate codec, as if the recording were compressed before upload.
    """
    return mp3_roundtrip(add_white_noise(band_pass(x, sr), snr_db, rng), sr, "32k")
