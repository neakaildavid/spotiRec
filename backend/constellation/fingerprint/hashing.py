"""Combinatorial hashing: pair each anchor peak with peaks in a target zone.

Why pairs instead of single peaks? A single peak carries only ~9 bits of
information (its frequency bin), so each one collides with millions of entries
in a large library and says almost nothing on its own. A pair ``(f1, f2, dt)``
carries ~26 bits, making it ~100,000x more specific, while still being
*translation invariant in time*: it only encodes the time *difference* between
the peaks, so the same pair produces the same hash wherever the clip starts.
The absolute time is kept separately as ``anchor_time`` for offset alignment.

Fan-out (pairing each anchor with several targets) adds redundancy: if noise
destroys one peak, the other pairs from the same anchor can still survive. The
cost is ``fan_out``x more rows in the index.
"""

from __future__ import annotations

import numpy as np

from constellation.config import FingerprintConfig


def pack_hash(f1: np.ndarray, f2: np.ndarray, dt: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """Pack ``(f1, f2, dt)`` into one integer: ``[f1 | f2 | dt]`` bit fields.

    One integer column is far cheaper to store and index than three, and lets
    lookups be a single ``WHERE hash = ANY(...)`` against a B-tree.
    """
    f1 = np.asarray(f1, np.int64)
    f2 = np.asarray(f2, np.int64)
    dt = np.asarray(dt, np.int64)
    return (f1 << (cfg.freq_bits + cfg.dt_bits)) | (f2 << cfg.dt_bits) | dt


def unpack_hash(h: np.ndarray, cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_hash` (used for tests and debugging)."""
    h = np.asarray(h, np.int64)
    fmask = (1 << cfg.freq_bits) - 1
    dmask = (1 << cfg.dt_bits) - 1
    return (h >> (cfg.freq_bits + cfg.dt_bits)) & fmask, (h >> cfg.dt_bits) & fmask, h & dmask


def generate_hashes(
    times: np.ndarray, freqs: np.ndarray, cfg: FingerprintConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Build hashes from a constellation map.

    For each anchor peak, the target zone is peaks with
    ``target_dt_min <= t2 - t1 <= target_dt_max`` and
    ``|f2 - f1| <= target_df_max``; the first ``fan_out`` of those (nearest in
    time) are paired with the anchor. Nearest-first matters: the closer the
    target, the more likely both peaks fall inside a short query clip.

    Returns:
        ``(hashes, anchor_times)``: int64 hashes and int32 anchor frame indices,
        one entry per pair.
    """
    order = np.lexsort((freqs, times))
    t = np.asarray(times, np.int64)[order]
    f = np.asarray(freqs, np.int64)[order]
    n = t.size
    if n < 2:
        return np.empty(0, np.int64), np.empty(0, np.int32)

    # Index range of candidate targets per anchor, found for all anchors at once.
    lo = np.searchsorted(t, t + cfg.target_dt_min, side="left")
    hi = np.searchsorted(t, t + cfg.target_dt_max, side="right")

    anchors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(n):
        if lo[i] >= hi[i]:
            continue
        cand = np.arange(lo[i], hi[i])
        cand = cand[np.abs(f[cand] - f[i]) <= cfg.target_df_max][: cfg.fan_out]
        if cand.size:
            anchors.append(np.full(cand.size, i))
            targets.append(cand)
    if not anchors:
        return np.empty(0, np.int64), np.empty(0, np.int32)

    a = np.concatenate(anchors)
    b = np.concatenate(targets)
    hashes = pack_hash(f[a], f[b], t[b] - t[a], cfg)
    return hashes, t[a].astype(np.int32)
