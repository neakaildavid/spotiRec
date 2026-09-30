"""Pydantic response models: the API's public contract (and OpenAPI docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from constellation.models import Song


class SongOut(BaseModel):
    id: int
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    duration_s: float
    source: str
    source_id: str | None
    audio_url: str = Field(description="Streams the audio; supports HTTP Range for seeking")
    created_at: datetime | None

    @classmethod
    def from_song(cls, s: Song) -> "SongOut":
        # file_path is deliberately not exposed: it's a server-side detail.
        return cls(
            id=s.id, title=s.title, artist=s.artist, album=s.album, genre=s.genre, duration_s=s.duration_s,
            source=s.source, source_id=s.source_id, audio_url=f"/songs/{s.id}/audio",
            created_at=s.created_at,
        )


class SongPage(BaseModel):
    items: list[SongOut]
    total: int
    limit: int
    offset: int


class QueryInfo(BaseModel):
    duration_s: float
    hashes: int
    # Constellation map of the query, [time_s, freq_hz], for the frontend animation.
    peaks: list[tuple[float, float]]


class Timing(BaseModel):
    decode_ms: float
    fingerprint_ms: float
    match_ms: float
    total_ms: float


class IdentifyResponse(BaseModel):
    match: bool
    song: SongOut | None = None
    offset_s: float | None = Field(None, description="Where in the song the clip starts")
    confidence: float = Field(description="0..1; a match requires the calibrated threshold")
    aligned_matches: int
    runner_up_matches: int
    query: QueryInfo
    timing: Timing


class IngestResponse(BaseModel):
    outcome: Literal["ingested", "updated", "skipped"]
    song: SongOut
    indexes: list[str]


class Health(BaseModel):
    status: Literal["ok"]
    songs: int
    fingerprint_version: str
    embedding_model: str | None = Field(description="Loaded model for audio/text discovery, if any")


class SimilarItem(BaseModel):
    song: SongOut
    score: float = Field(description="Cosine similarity of CLAP embeddings (higher = more alike)")


class DiscoverResponse(BaseModel):
    items: list[SimilarItem]
    model: str
    embed_ms: float = Field(description="Time in the embedding model (0 when vectors are precomputed)")
    search_ms: float


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=200, examples=["mellow acoustic guitar with soft vocals"])
    k: int = Field(10, ge=1, le=50)


class GenreCount(BaseModel):
    genre: str
    songs: int
