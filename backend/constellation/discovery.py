"""Discovery: "songs that sound like X", where X is a library song, an audio
clip, or a text description. Sits on top of the embedding index.

Kept out of the API layer so the same logic serves HTTP, the CLI and tests.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from constellation.embedding.base import Embedder, TextEmbedder
from constellation.models import Song
from constellation.storage.base import Storage


class DiscoveryUnavailable(RuntimeError):
    """The requested discovery mode needs a model that isn't loaded/installed."""


@dataclass(frozen=True)
class SimilarSong:
    song: Song
    score: float  # cosine similarity in [-1, 1]; CLAP neighbours typically 0.6-0.95


@dataclass(frozen=True)
class DiscoveryResult:
    items: list[SimilarSong]
    embed_ms: float  # time spent in the model (0 for library songs: precomputed)
    search_ms: float


class DiscoveryService:
    """Nearest-neighbour search with an artist-diversity cap.

    Args:
        storage: must implement the EmbeddingStore and SongStore protocols.
        model: embedding model name in ``song_embeddings`` (e.g. "clap-htsat").
        embedder: needed only for clip and text queries; song-to-song
            similarity uses stored vectors and works without loading a model.
        max_per_artist: raw nearest neighbours are often the rest of the query's
            album, which is accurate but a poor *discovery* list; capping per
            artist trades a little similarity for variety.
    """

    OVERFETCH = 4  # fetch k * OVERFETCH candidates so the artist cap can still fill k

    def __init__(self, storage: Storage, model: str, embedder: Embedder | None = None, max_per_artist: int = 2):
        self.storage = storage
        self.model = model
        self.embedder = embedder
        self.max_per_artist = max_per_artist
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache_size = 256
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------ queries

    def similar_to_song(self, song_id: int, k: int = 10) -> DiscoveryResult | None:
        """Songs that sound like a library song; None if it has no embedding yet."""
        vec = self.storage.get_embedding(song_id, self.model)
        if vec is None:
            return None
        song = self.storage.get_song(song_id)
        same_artist = _artist_key(song) if song else None
        t0 = time.perf_counter()
        items = self._search(vec, k, exclude=(song_id,), seed_artist=same_artist)
        return DiscoveryResult(items, 0.0, (time.perf_counter() - t0) * 1000)

    def similar_to_audio(self, samples: np.ndarray, k: int = 10) -> DiscoveryResult:
        """Songs that sound like a clip (e.g. one that identify couldn't match)."""
        if self.embedder is None:
            raise DiscoveryUnavailable("audio discovery needs the embedding model (pip install -e '.[embeddings]')")
        t0 = time.perf_counter()
        vec = self.embedder.embed_audio(samples)
        t1 = time.perf_counter()
        items = self._search(vec, k)
        return DiscoveryResult(items, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000)

    def search_text(self, query: str, k: int = 10) -> DiscoveryResult:
        """Songs matching a description, via CLAP's shared text/audio space."""
        if not isinstance(self.embedder, TextEmbedder):
            raise DiscoveryUnavailable("text search needs a text-capable embedding model (CLAP)")
        key = " ".join(query.lower().split())
        t0 = time.perf_counter()
        vec = self._cached_text(key)
        t1 = time.perf_counter()
        items = self._search(vec, k)
        return DiscoveryResult(items, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000)

    # ------------------------------------------------------------ internals

    def _cached_text(self, key: str) -> np.ndarray:
        with self._cache_lock:
            if key in self._text_cache:
                self._text_cache.move_to_end(key)
                return self._text_cache[key]
        vec = self.embedder.embed_text([key])[0]  # type: ignore[union-attr]
        with self._cache_lock:
            self._text_cache[key] = vec
            if len(self._text_cache) > self._text_cache_size:
                self._text_cache.popitem(last=False)
        return vec

    def _search(
        self, vec: np.ndarray, k: int, exclude: tuple[int, ...] = (), seed_artist: str | None = None
    ) -> list[SimilarSong]:
        hits = self.storage.nearest_embeddings(vec, self.model, k * self.OVERFETCH, exclude=exclude)
        songs = self.storage.get_songs([sid for sid, _ in hits])
        per_artist: dict[str, int] = {}
        if seed_artist:
            # The query song's own artist counts toward the cap, so "similar to
            # X" doesn't come back as mostly more X.
            per_artist[seed_artist] = 1
        out: list[SimilarSong] = []
        for sid, score in hits:
            song = songs.get(sid)
            if song is None:
                continue
            key = _artist_key(song)
            if key is not None:
                if per_artist.get(key, 0) >= self.max_per_artist:
                    continue
                per_artist[key] = per_artist.get(key, 0) + 1
            out.append(SimilarSong(song, score))
            if len(out) == k:
                break
        return out


def _artist_key(song: Song) -> str | None:
    """Normalized artist name; unknown artists are never capped."""
    return song.artist.strip().lower() if song.artist and song.artist.strip() else None
