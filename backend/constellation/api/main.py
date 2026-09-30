"""FastAPI application factory.

    uvicorn constellation.api.main:app --reload      # or: constellation serve

``create_app`` takes an optional storage so tests can inject ``MemoryStorage``;
by default the app opens a Postgres pool on startup and closes it on shutdown.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constellation.api.deps import get_storage
from constellation.api.routes import discover, identify, songs
from constellation.api.schemas import Health
from constellation.api.settings import Settings
from constellation.audio import AudioDecodeError
from constellation.config import FingerprintConfig, MatchConfig
from constellation.discovery import DiscoveryService, DiscoveryUnavailable
from constellation.embedding.base import Embedder
from constellation.embedding.clap import ClapEmbedder
from constellation.pipeline import EmbeddingIndexer, FingerprintIndexer, embeddings_available
from constellation.storage.base import Storage

log = logging.getLogger(__name__)
_DEFAULT = object()  # "choose automatically" sentinel (None means "no embedder")


def create_app(
    storage: Storage | None = None,
    settings: Settings | None = None,
    fp_config: FingerprintConfig | None = None,
    match_config: MatchConfig | None = None,
    indexers: list | None = None,
    embedder: Embedder | None | object = _DEFAULT,
) -> FastAPI:
    settings = settings or Settings.from_env()
    fp_config = fp_config or FingerprintConfig()
    if embedder is _DEFAULT:
        embedder = ClapEmbedder() if embeddings_available() else None
    # One embedder instance shared by ingestion (POST /songs) and discovery, so
    # the model is loaded into memory once.
    model_name = embedder.name if embedder is not None else ClapEmbedder.name  # type: ignore[union-attr]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = None
        if storage is None:
            from constellation.storage.postgres import PostgresStorage

            owned = PostgresStorage(settings.database_url)
            owned.init_schema()
            owned.warm_up()  # avoid a multi-second first query after restart
        app.state.storage = storage or owned
        app.state.discovery = DiscoveryService(app.state.storage, model_name, embedder)  # type: ignore[arg-type]
        if hasattr(embedder, "warm_up"):
            # Load the model in the background: the API serves identify/library
            # requests immediately; discovery calls wait on the model lock.
            def _warm() -> None:
                try:
                    embedder.warm_up()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001 - discovery will surface the error
                    log.exception("embedding model warm-up failed")

            threading.Thread(target=_warm, name="embedder-warm-up", daemon=True).start()
        try:
            yield
        finally:
            if owned is not None:
                owned.close()

    app = FastAPI(
        title="Constellation API",
        version="0.1.0",
        description="Shazam-style audio identification over a Creative Commons music library.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    # Queries must be fingerprinted with the same config as the index; the
    # version is exposed on /health so a mismatch is easy to spot.
    app.state.fp_config = fp_config
    app.state.match_config = match_config or MatchConfig()
    if indexers is None:
        indexers = [FingerprintIndexer(fp_config)]
        if embedder is not None:
            indexers.append(EmbeddingIndexer(embedder))  # type: ignore[arg-type]
    app.state.indexers = indexers

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    @app.exception_handler(AudioDecodeError)
    async def _decode_error(request: Request, exc: AudioDecodeError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "could not decode audio"})

    @app.exception_handler(DiscoveryUnavailable)
    async def _no_model(request: Request, exc: DiscoveryUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", response_model=Health, tags=["meta"])
    def health(storage: Storage = Depends(get_storage)) -> Health:
        return Health(
            status="ok",
            songs=storage.count_songs(),
            fingerprint_version=fp_config.version(),
            embedding_model=embedder.name if embedder is not None else None,  # type: ignore[union-attr]
        )

    app.include_router(identify.router)
    app.include_router(songs.router)
    app.include_router(discover.router)
    return app


app = create_app()
