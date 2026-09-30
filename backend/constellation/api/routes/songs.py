"""Library endpoints: list, add, fetch, random pick, and audio streaming."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from constellation.api.deps import get_settings, get_storage, read_upload, upload_suffix
from constellation.api.schemas import IngestResponse, SongOut, SongPage
from constellation.api.settings import Settings
from constellation.pipeline import Outcome, SongSource, process_song
from constellation.storage.base import Storage

router = APIRouter(prefix="/songs", tags=["songs"])


@router.get("", response_model=SongPage)
def list_songs(
    q: str | None = Query(None, max_length=200, description="Search title or artist"),
    sort: Literal["id", "title", "artist", "duration_s", "created_at"] = "id",
    genre: str | None = Query(None, max_length=100, description="Exact top-level genre"),
    order: Literal["asc", "desc"] = "asc",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    storage: Storage = Depends(get_storage),
) -> SongPage:
    songs = storage.list_songs(limit=limit, offset=offset, query=q, sort=sort, descending=order == "desc", genre=genre)
    return SongPage(
        items=[SongOut.from_song(s) for s in songs], total=storage.count_songs(q, genre), limit=limit, offset=offset
    )


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED,
             responses={200: {"description": "Identical audio was already in the library"}})
def add_song(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: str | None = Form(None, max_length=300),
    artist: str | None = Form(None, max_length=300),
    album: str | None = Form(None, max_length=300),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Ingest one song through the same pipeline as the batch CLI.

    The file is stored under its content hash, so re-uploading identical audio
    is a no-op that returns the existing song with HTTP 200.
    """
    data = read_upload(file, settings.max_song_bytes)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{hashlib.sha256(data).hexdigest()}{upload_suffix(file)}"
    created_file = not dest.exists()
    if created_file:
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    source = SongSource(
        path=dest,
        title=title or Path(file.filename or "").stem or None,
        artist=artist,
        album=album,
        source="upload",
    )
    try:
        result = process_song(storage, source, request.app.state.indexers)
    except Exception:
        if created_file:
            dest.unlink(missing_ok=True)
        raise

    song = result.song
    if song is None:  # pragma: no cover - process_song always returns the song
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "ingest returned no song")
    if result.outcome is Outcome.SKIPPED:
        response.status_code = status.HTTP_200_OK
        if created_file and Path(song.file_path) != dest.resolve():
            dest.unlink(missing_ok=True)  # same audio already stored elsewhere
    return IngestResponse(outcome=result.outcome.value, song=SongOut.from_song(song), indexes=result.indexes)


# Declared before /{song_id} so "random" isn't parsed as an id.
@router.get("/random", response_model=SongOut)
def random_song(storage: Storage = Depends(get_storage)) -> SongOut:
    song = storage.random_song()
    if song is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "library is empty")
    return SongOut.from_song(song)


@router.get("/{song_id}", response_model=SongOut)
def get_song(song_id: int, storage: Storage = Depends(get_storage)) -> SongOut:
    song = storage.get_song(song_id)
    if song is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "song not found")
    return SongOut.from_song(song)


@router.get("/{song_id}/audio", response_class=FileResponse)
def song_audio(song_id: int, storage: Storage = Depends(get_storage)) -> FileResponse:
    """Stream the song's audio. Starlette's FileResponse honours ``Range``
    headers (206 Partial Content), which is what lets the player seek straight
    to the matched offset without downloading the whole file first."""
    song = storage.get_song(song_id)
    if song is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "song not found")
    path = Path(song.file_path)  # comes from the DB, never from the request
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "audio file missing")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
