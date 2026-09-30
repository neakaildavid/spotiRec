"""FastAPI application factory.

    uvicorn constellation.api.main:app --reload      # or: constellation serve

``create_app`` takes an optional storage so tests can inject ``MemoryStorage``;
by default the app opens a Postgres pool on startup and closes it on shutdown.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constellation.api.deps import get_storage
from constellation.api.routes import identify, songs
from constellation.api.schemas import Health
from constellation.api.settings import Settings
from constellation.audio import AudioDecodeError
from constellation.config import FingerprintConfig, MatchConfig
from constellation.pipeline import default_indexers
from constellation.storage.base import Storage


def create_app(
    storage: Storage | None = None,
    settings: Settings | None = None,
    fp_config: FingerprintConfig | None = None,
    match_config: MatchConfig | None = None,
    indexers: list | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    fp_config = fp_config or FingerprintConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = None
        if storage is None:
            from constellation.storage.postgres import PostgresStorage

            owned = PostgresStorage(settings.database_url)
            owned.init_schema()
            owned.warm_up()  # avoid a multi-second first query after restart
        app.state.storage = storage or owned
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
    app.state.indexers = indexers if indexers is not None else default_indexers(fp_config)

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

    @app.get("/health", response_model=Health, tags=["meta"])
    def health(storage: Storage = Depends(get_storage)) -> Health:
        return Health(status="ok", songs=storage.count_songs(), fingerprint_version=fp_config.version())

    app.include_router(identify.router)
    app.include_router(songs.router)
    return app


app = create_app()
