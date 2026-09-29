"""Fingerprinting pipeline: samples -> spectrogram -> peaks -> hashes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.config import FingerprintConfig
from constellation.fingerprint.hashing import generate_hashes, pack_hash, unpack_hash
from constellation.fingerprint.peaks import find_peaks
from constellation.fingerprint.spectrogram import spectrogram


@dataclass(frozen=True)
class Fingerprint:
    """Everything derived from one piece of audio.

    ``hashes``/``anchor_times`` go into (or are looked up in) the index. The
    peaks are kept too so the API can return the query's constellation map for
    the frontend visualization.
    """

    hashes: np.ndarray  # int64, one per anchor/target pair
    anchor_times: np.ndarray  # int32 frame index of each hash's anchor
    peak_times: np.ndarray  # int32 frame indices
    peak_freqs: np.ndarray  # int32 bin indices

    def __len__(self) -> int:
        return int(self.hashes.size)


def fingerprint(samples: np.ndarray, cfg: FingerprintConfig | None = None) -> Fingerprint:
    """Fingerprint mono samples already at ``cfg.sample_rate``."""
    cfg = cfg or FingerprintConfig()
    peak_times, peak_freqs = find_peaks(spectrogram(samples, cfg), cfg)
    hashes, anchor_times = generate_hashes(peak_times, peak_freqs, cfg)
    return Fingerprint(hashes, anchor_times, peak_times, peak_freqs)


__all__ = [
    "Fingerprint",
    "find_peaks",
    "fingerprint",
    "generate_hashes",
    "pack_hash",
    "spectrogram",
    "unpack_hash",
]
