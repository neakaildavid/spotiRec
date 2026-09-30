"""In-memory Storage implementation, for unit tests and quick experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from constellation.models import IndexStatus, NewSong, Song
from constellation.similarity import top_k
from constellation.storage.base import SONG_SORT_FIELDS, LookupResult


class MemoryStorage:
    """Dict-backed store. Not thread-safe; ``transaction()`` does not roll back."""

    def __init__(self) -> None:
        self._songs: dict[int, Song] = {}
        self._by_content: dict[str, int] = {}
        self._next_id = 1
        # hash -> list of (song_id, anchor_time): an inverted index, like the
        # B-tree on fingerprints(hash) in Postgres.
        self._index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._status: dict[tuple[int, str], IndexStatus] = {}
        self._vectors: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    # --- SongStore ---
    def add_song(self, song: NewSong) -> Song:
        if song.content_hash in self._by_content:
            raise ValueError(f"duplicate content_hash {song.content_hash}")
        row = Song(**vars(song), id=self._next_id, created_at=datetime.now(timezone.utc))
        self._songs[row.id] = row
        self._by_content[row.content_hash] = row.id
        self._next_id += 1
        return row

    def get_song(self, song_id: int) -> Song | None:
        return self._songs.get(song_id)

    def get_song_by_content_hash(self, content_hash: str) -> Song | None:
        sid = self._by_content.get(content_hash)
        return self._songs.get(sid) if sid is not None else None

    def get_songs(self, song_ids: list[int]) -> dict[int, Song]:
        return {i: self._songs[i] for i in song_ids if i in self._songs}

    def _filtered(self, query: str | None, genre: str | None = None) -> list[Song]:
        songs = list(self._songs.values())
        if genre:
            songs = [s for s in songs if s.genre == genre]
        if query:
            q = query.lower()
            songs = [s for s in songs if q in (s.title or "").lower() or q in (s.artist or "").lower()]
        return songs

    def list_songs(
        self,
        limit: int = 100,
        offset: int = 0,
        query: str | None = None,
        sort: str = "id",
        descending: bool = False,
        genre: str | None = None,
    ) -> list[Song]:
        if sort not in SONG_SORT_FIELDS:
            raise ValueError(f"cannot sort by {sort!r}")
        songs = self._filtered(query, genre)
        # Nulls last in both directions, ties broken by id (matches Postgres).
        present = [s for s in songs if getattr(s, sort) is not None]
        missing = [s for s in songs if getattr(s, sort) is None]
        key = lambda s: (str(getattr(s, sort)).lower() if isinstance(getattr(s, sort), str) else getattr(s, sort), s.id)
        present.sort(key=key, reverse=descending)
        return (present + sorted(missing, key=lambda s: s.id))[offset : offset + limit]

    def count_songs(self, query: str | None = None, genre: str | None = None) -> int:
        return len(self._filtered(query, genre))

    def set_genres(self, genres: dict[int, str]) -> int:
        n = 0
        for sid, g in genres.items():
            if sid in self._songs:
                self._songs[sid] = replace(self._songs[sid], genre=g)
                n += 1
        return n

    def list_genres(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = defaultdict(int)
        for s in self._songs.values():
            if s.genre:
                counts[s.genre] += 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def random_song(self) -> Song | None:
        return random.choice(list(self._songs.values())) if self._songs else None

    # --- FingerprintStore ---
    def add_fingerprints(self, song_id: int, hashes: np.ndarray, anchor_times: np.ndarray) -> None:
        for h, t in zip(hashes.tolist(), anchor_times.tolist()):
            self._index[h].append((song_id, t))

    def delete_fingerprints(self, song_id: int) -> None:
        for h in list(self._index):
            self._index[h] = [e for e in self._index[h] if e[0] != song_id]

    def lookup(self, hashes: np.ndarray) -> LookupResult:
        out_h: list[int] = []
        out_s: list[int] = []
        out_t: list[int] = []
        for h in np.unique(hashes).tolist():
            for sid, t in self._index.get(h, ()):
                out_h.append(h)
                out_s.append(sid)
                out_t.append(t)
        if not out_h:
            return LookupResult.empty()
        return LookupResult(
            np.asarray(out_h, np.int64), np.asarray(out_s, np.int64), np.asarray(out_t, np.int64)
        )

    # --- EmbeddingStore (exact search; fine for tests and small libraries) ---
    def add_embedding(self, song_id: int, model: str, vec: np.ndarray) -> None:
        self._vectors[model][song_id] = np.asarray(vec, np.float32)

    def delete_embedding(self, song_id: int, model: str) -> None:
        self._vectors[model].pop(song_id, None)

    def get_embedding(self, song_id: int, model: str) -> np.ndarray | None:
        return self._vectors[model].get(song_id)

    def all_embeddings(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        vecs = self._vectors[model]
        if not vecs:
            return np.empty(0, np.int64), np.empty((0, 0), np.float32)
        return np.fromiter(vecs, np.int64), np.stack(list(vecs.values()))

    def nearest_embeddings(
        self, vec: np.ndarray, model: str, k: int, exclude: tuple[int, ...] = ()
    ) -> list[tuple[int, float]]:
        ids, mat = self.all_embeddings(model)
        if ids.size == 0:
            return []
        skip = np.flatnonzero(np.isin(ids, np.asarray(exclude, np.int64)))
        idx, scores = top_k(np.asarray(vec, np.float32), mat, k, exclude=skip)
        return [(int(ids[i]), float(s)) for i, s in zip(idx, scores)]

    # --- IndexStatusStore ---
    def get_index_status(self, song_id: int, index_name: str) -> IndexStatus | None:
        return self._status.get((song_id, index_name))

    def set_index_status(self, status: IndexStatus) -> None:
        if status.indexed_at is None:
            status = replace(status, indexed_at=datetime.now(timezone.utc))
        self._status[(status.song_id, status.index_name)] = status

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield
