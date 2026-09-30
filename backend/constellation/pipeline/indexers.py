"""Indexers: each one turns decoded audio into rows for one index.

The pipeline treats indexers uniformly, so Phase 2 adds an ``EmbeddingIndexer``
(e.g. 16 kHz audio -> model -> vectors -> pgvector) by implementing this
Protocol and appending it to the list in :func:`default_indexers`; the ingest
flow, idempotency tracking and CLI stay unchanged.

``compute`` is kept separate from ``write`` because they run in different
places: ``compute`` is pure CPU work on numpy arrays and runs in worker
processes, while ``write`` touches the database and runs in the single writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np

from constellation.config import FingerprintConfig
from constellation.embedding.base import Embedder
from constellation.fingerprint import fingerprint
from constellation.storage.base import Storage


class Indexer(Protocol):
    name: str
    # True if ``compute`` can run in many worker processes at once. False for
    # model-based indexers: every worker would load its own copy of the model
    # (~0.8 GB each), and one GPU process is faster anyway.
    parallel: bool

    @property
    def sample_rate(self) -> int:
        """Rate the audio must be decoded at for this indexer."""
        ...

    @property
    def version(self) -> str:
        """Changes whenever output for the same audio would change."""
        ...

    def compute(self, samples: np.ndarray) -> Any:
        """Pure function of the audio; must return something picklable."""
        ...

    def write(self, storage: Storage, song_id: int, payload: Any) -> int:
        """Store ``payload`` for ``song_id``; return the number of items written."""
        ...

    def delete(self, storage: Storage, song_id: int) -> None:
        """Remove this index's rows for ``song_id`` (before re-indexing)."""
        ...


@dataclass(frozen=True)
class FingerprintIndexer:
    """Shazam-style landmark hashes -> ``fingerprints`` table."""

    config: FingerprintConfig = field(default_factory=FingerprintConfig)
    name: ClassVar[str] = "fingerprint"
    parallel: ClassVar[bool] = True

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def version(self) -> str:
        return self.config.version()

    def compute(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fp = fingerprint(samples, self.config)
        return fp.hashes, fp.anchor_times

    def write(self, storage: Storage, song_id: int, payload: tuple[np.ndarray, np.ndarray]) -> int:
        hashes, anchor_times = payload
        storage.add_fingerprints(song_id, hashes, anchor_times)
        return int(hashes.size)

    def delete(self, storage: Storage, song_id: int) -> None:
        storage.delete_fingerprints(song_id)


@dataclass(frozen=True)
class EmbeddingIndexer:
    """Song-level "sounds like" vector -> ``song_embeddings`` (Phase 2).

    Wraps any :class:`~constellation.embedding.base.Embedder`; the index name
    includes the model so several embedding models can be indexed side by side.
    """

    embedder: Embedder
    parallel: ClassVar[bool] = False

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"embedding:{self.embedder.name}"

    @property
    def sample_rate(self) -> int:
        return self.embedder.sample_rate

    @property
    def version(self) -> str:
        return self.embedder.version

    def compute(self, samples: np.ndarray) -> np.ndarray:
        return self.embedder.embed_audio(samples)

    def write(self, storage: Storage, song_id: int, payload: np.ndarray) -> int:
        storage.add_embedding(song_id, self.embedder.name, payload)
        return 1

    def delete(self, storage: Storage, song_id: int) -> None:
        storage.delete_embedding(song_id, self.embedder.name)


def embeddings_available() -> bool:
    """True if the optional ``embeddings`` extra (torch + transformers) is installed."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def default_indexers(fp_config: FingerprintConfig | None = None, embedder: Embedder | None = None) -> list[Indexer]:
    """The indexes every song is processed into.

    Fingerprints always; CLAP embeddings when the ``embeddings`` extra is
    installed (or when an embedder is passed explicitly).
    """
    indexers: list[Indexer] = [FingerprintIndexer(fp_config or FingerprintConfig())]
    if embedder is None and embeddings_available():
        from constellation.embedding.clap import ClapEmbedder

        embedder = ClapEmbedder()
    if embedder is not None:
        indexers.append(EmbeddingIndexer(embedder))
    return indexers
