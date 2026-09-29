"""Short-time Fourier transform -> log-magnitude spectrogram."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

from constellation.config import FingerprintConfig

_EPS = 1e-10


def spectrogram(samples: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Return a log-magnitude spectrogram of shape ``(n_freq_bins, n_frames)`` in dB.

    Frame ``t`` covers samples ``[t*hop, t*hop + n_fft)``. We deliberately do
    *not* center/pad frames (unlike librosa's default), so frame indices map to
    absolute sample positions the same way for library songs and query clips;
    that keeps offset arithmetic in the matcher exact.

    Log magnitude (dB) is used because loudness perception and musical dynamics
    are roughly logarithmic: in linear magnitude a few loud bass notes would
    dwarf everything else and peak picking would ignore quieter melodic content.

    The Nyquist bin is dropped so there are exactly ``n_fft // 2`` bins, which
    fit in ``cfg.freq_bits`` when packing hashes.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("expected mono 1-D samples")
    if x.size < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - x.size))

    frames = sliding_window_view(x, cfg.n_fft)[:: cfg.hop_length]  # (n_frames, n_fft), a view
    # Hann window: tapering frame edges suppresses spectral leakage, so a pure
    # tone shows up as one sharp peak instead of smearing across many bins.
    window = get_window("hann", cfg.n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))[:, : cfg.n_freq_bins]
    return (20.0 * np.log10(mag + _EPS)).T.astype(np.float32)
