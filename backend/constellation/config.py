"""Tunable parameters for fingerprinting and matching.

All algorithm knobs live here as frozen dataclasses so that (a) the eval script
can sweep them, and (b) a fingerprint index can record exactly which config
produced it (see ``FingerprintConfig.version``). Hashes produced under one config
are meaningless under another, so the version is what makes re-indexing safe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FingerprintConfig:
    """Parameters for spectrogram -> peaks -> hashes.

    Defaults target ~11 kHz mono audio. At 11,025 Hz, ``n_fft=1024`` gives
    ~93 ms windows and ~10.8 Hz frequency bins; ``hop_length=256`` gives a
    frame every ~23 ms, which is also the time resolution of the returned
    match offset.
    """

    # --- Decoding / STFT ---
    sample_rate: int = 11_025
    n_fft: int = 1024
    hop_length: int = 256

    # --- Peak picking (constellation map) ---
    # Neighborhood of the 2D max filter, in (frequency bins, frames). A point is
    # a peak only if it is the loudest point in this box, which spreads peaks
    # evenly over the spectrogram instead of clustering them on the loudest note.
    peak_neighborhood_freq: int = 15
    peak_neighborhood_time: int = 15
    # Peaks must be within this many dB of the loudest point in the clip. Using a
    # threshold *relative* to the clip makes fingerprints volume invariant.
    peak_dynamic_range_db: float = 60.0
    # Clips whose loudest point is below this are treated as silence.
    silence_floor_db: float = -60.0
    # Keep at most this many (strongest) peaks per second. This bounds DB size
    # and, under noise, keeps the strong musical peaks and drops noise peaks.
    peaks_per_second: int = 15
    # Ignore bins below this frequency (phone mics roll off low frequencies).
    min_freq_hz: float = 0.0

    # --- Hashing (anchor -> target zone pairing) ---
    fan_out: int = 5  # target peaks paired with each anchor
    target_dt_min: int = 1  # frames
    target_dt_max: int = 63  # frames (~1.46 s)
    target_df_max: int = 127  # max |f2 - f1| in bins
    freq_bits: int = 9  # bits for each of f1 and f2 -> bins 0..511
    dt_bits: int = 8  # bits for dt -> 0..255

    def __post_init__(self) -> None:
        if self.n_fft // 2 > (1 << self.freq_bits):
            raise ValueError(
                f"n_fft={self.n_fft} yields {self.n_fft // 2} usable bins; "
                f"freq_bits={self.freq_bits} only encodes {1 << self.freq_bits}"
            )
        if not 0 < self.target_dt_min <= self.target_dt_max < (1 << self.dt_bits):
            raise ValueError("need 0 < target_dt_min <= target_dt_max < 2**dt_bits")
        if 2 * self.freq_bits + self.dt_bits > 31:
            # Hashes are stored as signed 32-bit INTEGER in Postgres.
            raise ValueError("packed hash must fit in 31 bits")

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_freq_bins(self) -> int:
        """Usable bins: we drop the Nyquist bin so bins fit in ``freq_bits``."""
        return self.n_fft // 2

    def frames_to_seconds(self, frames: float) -> float:
        return frames * self.hop_length / self.sample_rate

    def version(self) -> str:
        """Short stable hash of every parameter; changes iff hashes would change."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class MatchConfig:
    """Parameters for scoring candidates and deciding "match" vs "no match".

    ``min_confidence`` is calibrated by ``eval/run_eval.py``: the lowest value
    with zero false positives on held-out songs that are *not* in the library
    (0.25 on the FMA eval; see README "Evaluation").
    """

    # Offsets within this many frames of each other are counted together. A
    # query almost never starts exactly on the library's frame grid, so the same
    # peak can land one frame earlier/later and split a true bin in two.
    offset_tolerance_frames: int = 1
    # Minimum number of time-aligned hash matches to declare a match.
    min_aligned_matches: int = 5
    # Minimum confidence (0..1) to declare a match. Calibrated, see docstring.
    min_confidence: float = 0.25
    # Aligned-match count at which the "strength" part of confidence saturates.
    confidence_saturation: int = 25
