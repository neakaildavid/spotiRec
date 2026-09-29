"""FastAPI dependencies and request helpers."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile, status

from constellation.api.settings import Settings
from constellation.storage.base import Storage

_CONTENT_TYPE_SUFFIX = {
    "audio/webm": ".webm",  # Chrome/Firefox MediaRecorder
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",  # Safari MediaRecorder
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks, rejecting it as soon as it exceeds ``max_bytes``
    (rather than buffering an arbitrarily large body first)."""
    chunks, total = [], 0
    while chunk := file.file.read(1 << 20):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"file exceeds {max_bytes // 2**20} MB")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "empty file")
    return b"".join(chunks)


def upload_suffix(file: UploadFile) -> str:
    """Best-effort extension; ffmpeg probes the content anyway, this only helps it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and len(suffix) <= 6 and suffix[1:].isalnum():
        return suffix
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_SUFFIX.get(ctype) or mimetypes.guess_extension(ctype) or ""
