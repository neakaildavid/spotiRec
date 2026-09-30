"""Storage interfaces.

The rest of the app depends only on these Protocols, never on Postgres directly,
so the backing store can be swapped (in-memory for tests, Postgres in prod, or a
Redis fingerprint store later) without touching fingerprinting or matching code.

The fingerprint interface is deliberately tiny, ``add`` + ``lookup``: anything
that can do "multi-get by integer key" can serve as the index.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from constellation.models import IndexStatus, NewSong, Song


@dataclass(frozen=True)
class LookupResult:
    """Column-oriented result of a fingerprint lookup (one entry per DB row hit).

    Numpy columns instead of a list of tuples: a query can hit 10^5+ rows, and
    the matcher processes them with vectorized operations.
    """

    hashes: np.ndarray  # int64
    song_ids: np.ndarray  # int64
    anchor_times: np.ndarray  # int64, frames

    def __len__(self) -> int:
        return int(self.hashes.size)

    @classmethod
    def empty(cls) -> "LookupResult":
        z = np.empty(0, np.int64)
        return cls(z, z.copy(), z.copy())


SONG_SORT_FIELDS = ("id", "title", "artist", "duration_s", "created_at")


class SongStore(Protocol):
    def add_song(self, song: NewSong) -> Song: ...
    def get_song(self, song_id: int) -> Song | None: ...
    def get_song_by_content_hash(self, content_hash: str) -> Song | None: ...
    def get_songs(self, song_ids: list[int]) -> dict[int, Song]: ...
    def list_songs(
        self,
        limit: int = 100,
        offset: int = 0,
        query: str | None = None,
        sort: str = "id",
        descending: bool = False,
        genre: str | None = None,
    ) -> list[Song]:
        """Page through songs; ``query`` is a case-insensitive substring match on
        title/artist; ``sort`` is one of ``SONG_SORT_FIELDS``; ``genre`` is exact."""
        ...

    def count_songs(self, query: str | None = None, genre: str | None = None) -> int: ...
    def set_genres(self, genres: dict[int, str]) -> int:
        """Set ``genre`` for many songs at once (song_id -> genre); returns rows updated."""
        ...
    def list_genres(self) -> list[tuple[str, int]]:
        """``(genre, song count)`` pairs, most common first."""
        ...
    def random_song(self) -> Song | None: ...


class FingerprintStore(Protocol):
    def add_fingerprints(self, song_id: int, hashes: np.ndarray, anchor_times: np.ndarray) -> None: ...
    def delete_fingerprints(self, song_id: int) -> None: ...
    def lookup(self, hashes: np.ndarray) -> LookupResult: ...


class EmbeddingStore(Protocol):
    """One unit vector per (song, model) plus nearest-neighbour search.

    Kept separate from ``FingerprintStore`` on purpose: the two indexes answer
    different questions (exact recording vs. similar sound) and could live in
    different backends (e.g. fingerprints in Redis, vectors in pgvector).
    """

    def add_embedding(self, song_id: int, model: str, vec: np.ndarray) -> None: ...
    def delete_embedding(self, song_id: int, model: str) -> None: ...
    def get_embedding(self, song_id: int, model: str) -> np.ndarray | None: ...
    def all_embeddings(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """``(song_ids, matrix)`` for every stored vector of ``model`` (exact search, eval)."""
        ...
    def nearest_embeddings(
        self, vec: np.ndarray, model: str, k: int, exclude: tuple[int, ...] = ()
    ) -> list[tuple[int, float]]:
        """``[(song_id, cosine similarity)]``, most similar first; may be approximate."""
        ...


class IndexStatusStore(Protocol):
    def get_index_status(self, song_id: int, index_name: str) -> IndexStatus | None: ...
    def set_index_status(self, status: IndexStatus) -> None: ...


class Storage(SongStore, FingerprintStore, EmbeddingStore, IndexStatusStore, Protocol):
    """Everything the ingestion pipeline needs, plus a unit of work.

    ``transaction()`` groups writes so that a song's fingerprints and its
    ``IndexStatus`` row commit together. A crash mid-song therefore leaves no
    partial index behind, which is what makes re-running ingestion safe.
    """

    def transaction(self) -> AbstractContextManager[None]: ...
