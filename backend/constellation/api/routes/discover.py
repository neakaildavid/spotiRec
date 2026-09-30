"""Discovery endpoints (Phase 2): similar songs, "sounds like" a clip, text search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from constellation.api.deps import get_settings, get_storage, read_upload, upload_suffix
from constellation.api.schemas import DiscoverResponse, GenreCount, SimilarItem, SongOut, TextQuery
from constellation.api.settings import Settings
from constellation.audio import load_audio_bytes
from constellation.discovery import DiscoveryResult, DiscoveryService
from constellation.storage.base import Storage

router = APIRouter(tags=["discover"])


def get_discovery(request: Request) -> DiscoveryService:
    return request.app.state.discovery


def _response(result: DiscoveryResult, svc: DiscoveryService) -> DiscoverResponse:
    return DiscoverResponse(
        items=[SimilarItem(song=SongOut.from_song(i.song), score=round(i.score, 4)) for i in result.items],
        model=svc.model,
        embed_ms=round(result.embed_ms, 1),
        search_ms=round(result.search_ms, 1),
    )


@router.get("/songs/{song_id}/similar", response_model=DiscoverResponse)
def similar_songs(
    song_id: int,
    k: int = Query(10, ge=1, le=50),
    storage: Storage = Depends(get_storage),
    svc: DiscoveryService = Depends(get_discovery),
) -> DiscoverResponse:
    """Songs that sound like a library song. Uses precomputed vectors, so no model
    inference happens at request time (and it works without the model installed)."""
    if storage.get_song(song_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "song not found")
    result = svc.similar_to_song(song_id, k)
    if result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "song has no embedding yet (run: constellation ingest --indexes embedding)")
    return _response(result, svc)


@router.post("/discover/audio", response_model=DiscoverResponse)
def discover_by_audio(
    file: UploadFile = File(..., description="Audio clip; e.g. one /identify couldn't match"),
    k: int = Query(10, ge=1, le=50),
    settings: Settings = Depends(get_settings),
    svc: DiscoveryService = Depends(get_discovery),
) -> DiscoverResponse:
    """Library songs that *sound like* the clip. Unlike /identify this never says
    "no match": it answers "what's closest in sound", even for unknown songs."""
    if svc.embedder is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "embedding model not installed")
    data = read_upload(file, settings.max_identify_bytes)
    sr = svc.embedder.sample_rate
    samples = load_audio_bytes(data, sr, suffix=upload_suffix(file))[: int(settings.max_discover_seconds * sr)]
    if samples.size < settings.min_query_seconds * sr:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "clip too short")
    return _response(svc.similar_to_audio(samples, k), svc)


@router.post("/discover/text", response_model=DiscoverResponse)
def discover_by_text(body: TextQuery, svc: DiscoveryService = Depends(get_discovery)) -> DiscoverResponse:
    """Describe a sound ("upbeat electronic with a heavy bassline") and get songs.

    Works because CLAP embeds text and audio into the same space. Query
    embeddings are LRU-cached, so repeated searches skip the model.
    """
    return _response(svc.search_text(body.query, body.k), svc)


@router.get("/genres", response_model=list[GenreCount], tags=["songs"])
def genres(storage: Storage = Depends(get_storage)) -> list[GenreCount]:
    return [GenreCount(genre=g, songs=n) for g, n in storage.list_genres()]
