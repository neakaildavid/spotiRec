"""Decode any audio format to a mono float32 numpy array via ffmpeg.

We shell out to ffmpeg instead of using a Python decoder because it handles every
container/codec a user might upload (mp3, m4a, webm/opus from the browser's
MediaRecorder, flac, wav, ...) and resamples + downmixes in one fast native pass.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode the input or it contains no audio."""


def _ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise AudioDecodeError("ffmpeg not found on PATH")
    return path


def load_audio(
    path: str | os.PathLike,
    sample_rate: int = 11_025,
    *,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> np.ndarray:
    """Decode ``path`` to mono float32 samples in [-1, 1] at ``sample_rate``.

    Args:
        path: Any file ffmpeg can read.
        sample_rate: Output rate. Fingerprinting uses ~11 kHz: the musically
            useful peaks for matching sit below ~5 kHz, and halving the rate
            halves the STFT cost. Other indexers (Phase 2 embeddings) can ask
            for a different rate.
        start_s: Optional seek position (seconds) before decoding.
        duration_s: Optional maximum duration (seconds) to decode.

    Stereo is downmixed with ffmpeg's ``-ac 1`` matrix (each channel x 1/sqrt(2),
    i.e. power-preserving rather than a plain average), so levels can exceed
    1.0 for correlated loud stereo. Fingerprinting is volume invariant, so this
    doesn't matter for matching.
    """
    cmd = [_ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip() or "unknown error"
        raise AudioDecodeError(f"ffmpeg failed on {path}: {msg}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size == 0:
        raise AudioDecodeError(f"no audio decoded from {path}")
    return samples.astype(np.float32, copy=True)


def load_audio_bytes(data: bytes, sample_rate: int = 11_025, suffix: str = "") -> np.ndarray:
    """Decode an in-memory upload.

    Writes to a temp file rather than piping to ffmpeg's stdin: some containers
    (e.g. mp4/m4a with the index at the end) need a seekable input.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return load_audio(Path(tmp.name), sample_rate)
