"""API settings, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_list(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str | None = None  # None -> storage.postgres.database_url()
    upload_dir: Path = Path("data/uploads")
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    max_identify_bytes: int = 15 * 2**20
    max_song_bytes: int = 60 * 2**20
    # Only the first N seconds of a query are fingerprinted: longer clips add
    # latency without improving accuracy (10 s clips already score ~99%).
    max_query_seconds: float = 20.0
    min_query_seconds: float = 1.0
    # Discovery clips: CLAP embeds 10 s windows, so 30 s = 3 windows (~0.25 s on an M2).
    max_discover_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ.get("CONSTELLATION_DATABASE_URL"),
            upload_dir=Path(os.environ.get("CONSTELLATION_UPLOAD_DIR", "data/uploads")),
            cors_origins=_env_list("CONSTELLATION_CORS_ORIGINS", "http://localhost:5173"),
        )
