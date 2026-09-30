"""Plain data objects shared across storage, pipeline, matching and API layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NewSong:
    """A song about to be inserted (no id yet)."""

    title: str | None
    artist: str | None
    album: str | None
    duration_s: float
    file_path: str
    content_hash: str
    source: str = "fma"
    source_id: str | None = None
    genre: str | None = None


@dataclass(frozen=True)
class Song(NewSong):
    """A song row in the library."""

    id: int = 0
    created_at: datetime | None = None


@dataclass(frozen=True)
class IndexStatus:
    """Records that ``song_id`` was processed into ``index_name`` under ``version``."""

    song_id: int
    index_name: str
    version: str
    item_count: int
    indexed_at: datetime | None = None
