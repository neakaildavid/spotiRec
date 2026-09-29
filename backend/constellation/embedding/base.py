"""Audio embedders: map audio (and optionally text) to unit vectors whose cosine
similarity reflects how alike the music *sounds*.

Unlike fingerprints (which match one exact recording), embeddings capture
timbre, instrumentation, rhythm and mood, so two different songs can be close.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str  # stored as song_embeddings.model, e.g. "clap-htsat"
    dim: int
    sample_rate: int

    @property
    def version(self) -> str:
        """Changes whenever output for the same audio would change (model, windowing, pooling)."""
        ...

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        """Mono float32 samples at ``sample_rate`` -> unit vector of shape ``(dim,)``."""
        ...


@runtime_checkable
class TextEmbedder(Embedder, Protocol):
    """An embedder whose text and audio vectors live in the same space (CLAP)."""

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """-> unit vectors of shape ``(len(texts), dim)``."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale to unit length so that dot product == cosine similarity."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def split_windows(
    samples: np.ndarray, sample_rate: int, window_s: float = 10.0, max_windows: int = 12, min_last_s: float = 3.0
) -> list[np.ndarray]:
    """Cut audio into non-overlapping windows of ``window_s`` seconds.

    * A trailing partial window is kept only if it's at least ``min_last_s``
      long (a 1 s tail would get as much weight as a full window).
    * Audio shorter than one window yields a single (short) window.
    * Long inputs (a full-length upload) are capped at ``max_windows`` windows
      spread evenly across the track, bounding compute per song.
    """
    win = int(window_s * sample_rate)
    n = samples.size
    if n <= win:
        return [samples]
    starts = list(range(0, n - win + 1, win))
    tail = n - (starts[-1] + win)
    if tail >= min_last_s * sample_rate:
        starts.append(n - tail)
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    return [samples[s : s + win] for s in starts]
