"""Match a query fingerprint against the index using offset histograms.

Why an offset histogram? Looking up the query's hashes returns hits from many
songs: common hashes occur everywhere, so a raw "count of shared hashes" is
noisy. But for the *true* song every genuine hit satisfies

    db_anchor_time - query_anchor_time = (where the clip starts in the song)

i.e. all true hits agree on a single offset, while chance hits scatter across
random offsets. So for each song we histogram ``db_t - query_t`` and take the
tallest bin: that count measures *time-coherent* agreement, which random
collisions almost never produce. The bin's position is the answer to "where in
the song is this clip", for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellation.config import FingerprintConfig, MatchConfig
from constellation.fingerprint import Fingerprint
from constellation.storage.base import FingerprintStore, LookupResult

_OFFSET_SHIFT = np.int64(1 << 31)  # makes offsets non-negative for key packing


@dataclass(frozen=True)
class Candidate:
    """Best offset bin for one song."""

    song_id: int
    offset_frames: int
    aligned_matches: int


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one query. ``is_match`` is False for "no match".

    Scores are populated even for non-matches so the eval can calibrate
    thresholds from their distribution.
    """

    is_match: bool
    song_id: int | None
    offset_frames: int
    offset_s: float
    confidence: float
    aligned_matches: int  # height of the winning offset bin (with tolerance)
    runner_up_matches: int  # best bin of any *other* song: an estimate of chance agreement
    query_hashes: int
    db_hits: int
    candidates: tuple[Candidate, ...] = ()


def join_offsets(
    query_hashes: np.ndarray, query_times: np.ndarray, hits: LookupResult
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every DB hit with every query occurrence of the same hash.

    Returns ``(song_ids, offsets)`` with ``offset = db_anchor_time - query_anchor_time``.
    Fully vectorized (sort + searchsorted + repeat) because a query against a big
    library can produce 10^5+ hits and a Python loop would dominate latency.
    """
    if len(hits) == 0 or query_hashes.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    order = np.argsort(query_hashes, kind="stable")
    qh = np.asarray(query_hashes, np.int64)[order]
    qt = np.asarray(query_times, np.int64)[order]

    left = np.searchsorted(qh, hits.hashes, side="left")
    counts = np.searchsorted(qh, hits.hashes, side="right") - left
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    hit_idx = np.repeat(np.arange(len(hits)), counts)
    # For hit i, query indices left[i] .. left[i]+counts[i]-1.
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    q_idx = np.repeat(left, counts) + (np.arange(total) - run_start)

    offsets = hits.anchor_times[hit_idx] - qt[q_idx]
    return hits.song_ids[hit_idx], offsets


def score_candidates(
    song_ids: np.ndarray, offsets: np.ndarray, cfg: MatchConfig, top_k: int = 5
) -> list[Candidate]:
    """Build per-song offset histograms and return the best ``top_k`` songs.

    Each (song, offset) pair is packed into one int64 key so a single
    ``np.unique`` builds every song's histogram at once. The score of a bin is
    the count within ``+/- offset_tolerance_frames``, since a query that is not
    aligned to the library's frame grid can split one true bin across two.
    """
    if song_ids.size == 0:
        return []
    keys = (song_ids.astype(np.int64) << 32) | (offsets.astype(np.int64) + _OFFSET_SHIFT)
    uniq, counts = np.unique(keys, return_counts=True)

    w = cfg.offset_tolerance_frames
    csum = np.concatenate(([0], np.cumsum(counts)))
    lo = np.searchsorted(uniq, uniq - w, side="left")
    hi = np.searchsorted(uniq, uniq + w, side="right")
    scores = csum[hi] - csum[lo]

    # Keys are sorted by song first, so each song is one contiguous run.
    songs = uniq >> 32
    starts = np.flatnonzero(np.concatenate(([True], songs[1:] != songs[:-1])))
    best_per_song = np.maximum.reduceat(scores, starts)

    ranked = np.argsort(-best_per_song, kind="stable")[:top_k]
    out: list[Candidate] = []
    for g in ranked:
        s, e = starts[g], (starts[g + 1] if g + 1 < starts.size else uniq.size)
        i = s + int(np.argmax(scores[s:e]))
        out.append(
            Candidate(
                song_id=int(songs[i]),
                offset_frames=int((uniq[i] & 0xFFFFFFFF) - _OFFSET_SHIFT),
                aligned_matches=int(scores[i]),
            )
        )
    return out


def confidence_score(best: int, runner_up: int, cfg: MatchConfig) -> float:
    """Map raw counts to [0, 1].

    ``margin = 1 - runner_up/best``: how far the winner stands above the best
    *wrong* song, which estimates the chance-agreement background for this
    query. ``strength = min(1, best/saturation)``: a 3-vs-0 win has a perfect
    margin but is weak evidence. Confidence is their product.
    """
    if best <= 0:
        return 0.0
    margin = 1.0 - runner_up / best
    strength = min(1.0, best / cfg.confidence_saturation)
    return float(max(0.0, margin) * strength)


def match_fingerprint(
    fp: Fingerprint,
    store: FingerprintStore,
    fp_cfg: FingerprintConfig | None = None,
    match_cfg: MatchConfig | None = None,
) -> MatchResult:
    """Identify the song a query fingerprint came from."""
    fp_cfg = fp_cfg or FingerprintConfig()
    match_cfg = match_cfg or MatchConfig()

    hits = store.lookup(fp.hashes) if len(fp) else LookupResult.empty()
    song_ids, offsets = join_offsets(fp.hashes, fp.anchor_times, hits)
    cands = score_candidates(song_ids, offsets, match_cfg)

    if not cands:
        return MatchResult(False, None, 0, 0.0, 0.0, 0, 0, len(fp), len(hits))

    best = cands[0]
    runner_up = cands[1].aligned_matches if len(cands) > 1 else 0
    conf = confidence_score(best.aligned_matches, runner_up, match_cfg)
    is_match = best.aligned_matches >= match_cfg.min_aligned_matches and conf >= match_cfg.min_confidence
    return MatchResult(
        is_match=is_match,
        song_id=best.song_id if is_match else None,
        offset_frames=best.offset_frames,
        offset_s=fp_cfg.frames_to_seconds(best.offset_frames),
        confidence=conf,
        aligned_matches=best.aligned_matches,
        runner_up_matches=runner_up,
        query_hashes=len(fp),
        db_hits=len(hits),
        candidates=tuple(cands),
    )
