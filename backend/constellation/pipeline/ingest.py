"""Process songs into every index: the one place ingestion logic lives.

Each song goes through three phases:

1. **plan** (main process, cheap): hash the file, look it up, and decide which
   indexers still need to run. A song with every index present at the current
   version is skipped *before* decoding, so re-running over a 10k-song folder
   costs one sha256 + one query per file.
2. **compute** (worker processes, CPU heavy): decode once per required sample
   rate and run each indexer's ``compute``. No DB access, so it parallelizes
   freely across processes (numpy work here is largely GIL-bound Python glue,
   so threads wouldn't help much).
3. **commit** (main process, one transaction per song): insert the song if new,
   replace stale index rows, write new rows and the ``song_indexes`` markers.
   Because markers commit atomically with the rows, an interrupted run leaves
   either a fully indexed song or nothing, and the next run picks up cleanly.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from constellation.audio import load_audio
from constellation.models import IndexStatus, NewSong, Song
from constellation.pipeline.indexers import Indexer
from constellation.storage.base import Storage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SongSource:
    """A file to ingest plus whatever metadata we know about it."""

    path: Path
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    source: str = "upload"
    source_id: str | None = None


class Outcome(str, Enum):
    INGESTED = "ingested"  # new song
    UPDATED = "updated"  # existing song, some index (re)built
    SKIPPED = "skipped"  # everything already current
    FAILED = "failed"


@dataclass
class IngestResult:
    source: SongSource
    outcome: Outcome
    song: Song | None = None
    indexes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _Plan:
    source: SongSource
    content_hash: str
    indexers: list[Indexer]  # the ones that need to run


@dataclass
class _Computed:
    duration_s: float
    payloads: dict[str, Any]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _needs(storage: Storage, song: Song | None, indexers: list[Indexer]) -> list[Indexer]:
    """Indexers missing, or present at an outdated version, for ``song``."""
    if song is None:
        return list(indexers)
    out = []
    for ix in indexers:
        status = storage.get_index_status(song.id, ix.name)
        if status is None or status.version != ix.version:
            out.append(ix)
    return out


def plan_song(storage: Storage, source: SongSource, indexers: list[Indexer]) -> _Plan:
    content_hash = file_sha256(source.path)
    song = storage.get_song_by_content_hash(content_hash)
    return _Plan(source, content_hash, _needs(storage, song, indexers))


def compute_song(path: Path, indexers: list[Indexer]) -> _Computed:
    """Decode (once per distinct sample rate) and run every indexer's compute.

    Module-level and DB-free so it can run in a ``ProcessPoolExecutor``.
    """
    decoded: dict[int, Any] = {}
    payloads: dict[str, Any] = {}
    duration_s = 0.0
    for ix in indexers:
        sr = ix.sample_rate
        if sr not in decoded:
            decoded[sr] = load_audio(path, sr)
            duration_s = decoded[sr].size / sr
        payloads[ix.name] = ix.compute(decoded[sr])
    return _Computed(duration_s, payloads)


def commit_song(storage: Storage, plan: _Plan, computed: _Computed) -> IngestResult:
    """Write one song's rows for every planned indexer in a single transaction."""
    src = plan.source
    by_name = {ix.name: ix for ix in plan.indexers}
    with storage.transaction():
        # Re-check inside the transaction: another file in the same batch (or
        # another process) may have inserted identical content since planning.
        song = storage.get_song_by_content_hash(plan.content_hash)
        is_new = song is None
        if is_new:
            song = storage.add_song(
                NewSong(
                    title=src.title or src.path.stem,
                    artist=src.artist,
                    album=src.album,
                    duration_s=computed.duration_s,
                    file_path=str(src.path.resolve()),
                    content_hash=plan.content_hash,
                    source=src.source,
                    source_id=src.source_id,
                )
            )
        written = []
        for ix in _needs(storage, None if is_new else song, list(by_name.values())):
            if not is_new:
                ix.delete(storage, song.id)  # drop stale rows from an older version
            count = ix.write(storage, song.id, computed.payloads[ix.name])
            storage.set_index_status(IndexStatus(song.id, ix.name, ix.version, count))
            written.append(ix.name)
    outcome = Outcome.INGESTED if is_new else (Outcome.UPDATED if written else Outcome.SKIPPED)
    return IngestResult(src, outcome, song, written)


def process_song(storage: Storage, source: SongSource, indexers: list[Indexer]) -> IngestResult:
    """Ingest one song synchronously (used by ``POST /songs``)."""
    plan = plan_song(storage, source, indexers)
    if not plan.indexers:
        return IngestResult(source, Outcome.SKIPPED, storage.get_song_by_content_hash(plan.content_hash))
    return commit_song(storage, plan, compute_song(source.path, plan.indexers))


def ingest_many(
    storage: Storage,
    sources: Iterable[SongSource],
    indexers: list[Indexer],
    workers: int = 1,
    on_result: Callable[[IngestResult], None] | None = None,
) -> dict[Outcome, int]:
    """Batch ingestion with parallel compute and a single DB writer.

    Per-song failures (e.g. a corrupt mp3) are reported and skipped; nothing is
    recorded for them, so a later run retries them.
    """
    counts = {o: 0 for o in Outcome}

    def emit(r: IngestResult) -> None:
        counts[r.outcome] += 1
        if r.outcome is Outcome.FAILED:
            log.warning("failed %s: %s", r.source.path, r.error)
        if on_result:
            on_result(r)

    plans: list[_Plan] = []
    for src in sources:
        try:
            plan = plan_song(storage, src, indexers)
        except OSError as e:
            emit(IngestResult(src, Outcome.FAILED, error=str(e)))
            continue
        if plan.indexers:
            plans.append(plan)
        else:
            emit(IngestResult(src, Outcome.SKIPPED))

    def finish(plan: _Plan, get: Callable[[], _Computed]) -> None:
        try:
            emit(commit_song(storage, plan, get()))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            emit(IngestResult(plan.source, Outcome.FAILED, error=f"{type(e).__name__}: {e}"))

    if workers <= 1:
        for plan in plans:
            finish(plan, lambda p=plan: compute_song(p.source.path, p.indexers))
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(compute_song, p.source.path, p.indexers): p for p in plans}
            for fut in as_completed(futures):
                finish(futures[fut], fut.result)
    return counts
