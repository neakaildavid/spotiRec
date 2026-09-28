"""Shared fixtures: synthetic "music" so tests need no dataset."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from constellation.config import FingerprintConfig
from constellation.fingerprint import fingerprint
from constellation.models import NewSong
from constellation.storage import MemoryStorage

SR = 11_025


def synth_song(seed: int, duration_s: float = 30.0, sr: int = SR) -> np.ndarray:
    """Random note sequence with harmonics and attack/decay envelopes.

    Crude, but it has what matters for fingerprinting: sparse, time-varying
    spectral peaks that differ from song to song.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    out = np.zeros(n, np.float32)
    t0 = 0
    while t0 < n:
        dur = int(rng.uniform(0.12, 0.5) * sr)
        t = np.arange(dur) / sr
        env = np.minimum(1.0, t / 0.01) * np.exp(-t * rng.uniform(2, 8))
        for _ in range(rng.integers(1, 4)):  # 1-3 simultaneous notes
            f0 = 440.0 * 2 ** ((rng.integers(40, 90) - 69) / 12)
            amp = rng.uniform(0.2, 1.0)
            for k, hamp in enumerate((1.0, 0.5, 0.25), start=1):  # harmonics
                if f0 * k < sr / 2:
                    tone = amp * hamp * env * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
                    end = min(n, t0 + dur)
                    out[t0:end] += tone[: end - t0].astype(np.float32)
        t0 += dur
    out += 0.005 * rng.standard_normal(n).astype(np.float32)
    return out / np.abs(out).max() * 0.9


def add_white_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise so that signal power / noise power = ``snr_db``."""
    rng = np.random.default_rng(seed)
    p_signal = float(np.mean(x.astype(np.float64) ** 2))
    p_noise = p_signal / (10 ** (snr_db / 10))
    return (x + rng.standard_normal(x.size) * np.sqrt(p_noise)).astype(np.float32)


@pytest.fixture(scope="session")
def fp_cfg() -> FingerprintConfig:
    return FingerprintConfig()


@pytest.fixture(scope="session")
def library_audio() -> dict[int, np.ndarray]:
    """20 synthetic 30 s songs keyed by seed."""
    return {seed: synth_song(seed) for seed in range(20)}


@pytest.fixture(scope="session")
def library(library_audio, fp_cfg):
    """MemoryStorage with every synthetic song fingerprinted in.

    Returns ``(store, seed_to_song_id)``.
    """
    store = MemoryStorage()
    ids: dict[int, int] = {}
    for seed, audio in library_audio.items():
        song = store.add_song(
            NewSong(
                title=f"Song {seed}",
                artist="Synth",
                album=None,
                duration_s=audio.size / SR,
                file_path=f"synthetic/{seed}.wav",
                content_hash=hashlib.sha256(audio.tobytes()).hexdigest(),
            )
        )
        fp = fingerprint(audio, fp_cfg)
        store.add_fingerprints(song.id, fp.hashes, fp.anchor_times)
        ids[seed] = song.id
    return store, ids
