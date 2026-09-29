"""Peak picking: spectrogram -> sparse "constellation map" of (time, freq) points."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from constellation.config import FingerprintConfig


def find_peaks(spec_db: np.ndarray, cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Extract spectral peaks from a ``(freq, time)`` dB spectrogram.

    Why peaks? Local maxima are the most robust feature of a spectrogram: additive
    noise raises the floor and an equalizer tilts the spectrum, but the loudest
    point in a neighborhood usually stays the loudest point. Discarding
    everything else also turns a dense matrix into a few points per second,
    which is what makes an index over millions of songs tractable.

    Steps:
      1. 2D max filter: a point is a candidate if it equals the max of its
         ``(peak_neighborhood_freq x peak_neighborhood_time)`` box (and the box
         isn't flat).
      2. Amplitude threshold relative to the clip's loudest point (volume
         invariant), plus a silence check.
      3. Density cap: keep the ``peaks_per_second`` strongest per 1-second block,
         so every second of audio contributes and the DB size is predictable.

    Returns:
        ``(times, freqs)`` as int32 arrays (frame index, bin index), sorted by
        time then frequency.
    """
    empty = (np.empty(0, np.int32), np.empty(0, np.int32))
    if spec_db.size == 0:
        return empty

    spec = spec_db.copy()
    min_bin = int(np.ceil(cfg.min_freq_hz * cfg.n_fft / cfg.sample_rate))
    if min_bin > 0:
        spec[:min_bin] = -np.inf

    loudest = float(spec.max())
    if loudest < cfg.silence_floor_db:
        return empty

    size = (cfg.peak_neighborhood_freq, cfg.peak_neighborhood_time)
    neighborhood_max = maximum_filter(spec, size=size, mode="constant", cval=-np.inf)
    # A flat region (e.g. a zeroed band after codec low-passing) makes every
    # point "equal to its neighborhood max"; requiring the point to also exceed
    # its neighborhood min rejects plateaus.
    neighborhood_min = minimum_filter(spec, size=size, mode="nearest")
    is_peak = (
        (spec == neighborhood_max)
        & (spec > neighborhood_min)
        & (spec >= loudest - cfg.peak_dynamic_range_db)
    )
    freqs, times = np.nonzero(is_peak)
    if freqs.size == 0:
        return empty
    amps = spec[freqs, times]

    # Density cap: rank peaks by amplitude within each ~1 s block, keep top-k.
    frames_per_block = max(1, round(cfg.frames_per_second))
    block = times // frames_per_block
    order = np.lexsort((-amps, block))  # sort by block, then loudest first
    block_sorted = block[order]
    rank_in_block = np.arange(order.size) - np.searchsorted(block_sorted, block_sorted, side="left")
    keep = order[rank_in_block < cfg.peaks_per_second]

    times, freqs = times[keep], freqs[keep]
    final = np.lexsort((freqs, times))
    return times[final].astype(np.int32), freqs[final].astype(np.int32)
