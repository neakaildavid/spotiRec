"""Nearest-neighbour search over unit vectors (exact, numpy).

Used by the in-memory store and the eval. With unit vectors, cosine similarity
is a plain dot product, so exact search over n songs is one matrix-vector
product: ~1 ms for 10^4 songs x 512 dims. Approximate indexes (HNSW in
pgvector) only start paying off well beyond that.
"""

from __future__ import annotations

import numpy as np


def top_k(query: np.ndarray, matrix: np.ndarray, k: int, exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine scores of the ``k`` rows of ``matrix`` closest to ``query``.

    ``exclude``: row indices to skip (e.g. the query song itself).
    Uses ``argpartition`` (O(n)) and only sorts the k winners.
    """
    scores = matrix @ query
    if exclude is not None and exclude.size:
        scores = scores.copy()
        scores[exclude] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx, scores[idx].astype(np.float32)
