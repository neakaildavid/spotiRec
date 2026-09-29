"""Read Free Music Archive (FMA) metadata.

FMA's ``tracks.csv`` has a two-level header (e.g. ``track,title`` and
``artist,name``) plus a ``track_id`` row, so we parse it with the stdlib ``csv``
module and address columns by ``(group, field)`` instead of pulling in pandas.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_TRACK_FILE_RE = re.compile(r"^(\d{6})\.mp3$")


@dataclass(frozen=True)
class FmaTrack:
    track_id: int
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    subset: str | None  # "small" | "medium" | "large"
    duration_s: float | None


def track_id_from_path(path: str | Path) -> int | None:
    """``.../fma_small/000/000002.mp3`` -> ``2``; None if not an FMA file name."""
    m = _TRACK_FILE_RE.match(Path(path).name)
    return int(m.group(1)) if m else None


def audio_relpath(track_id: int) -> str:
    """FMA's on-disk layout: files sharded into folders by the first 3 digits."""
    tid = f"{track_id:06d}"
    return f"{tid[:3]}/{tid}.mp3"


def load_tracks(tracks_csv: str | Path) -> dict[int, FmaTrack]:
    """Parse ``tracks.csv`` into ``{track_id: FmaTrack}``."""
    csv.field_size_limit(sys.maxsize)  # some artist bios are huge
    with open(tracks_csv, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        groups = next(reader)
        fields = next(reader)
        next(reader)  # "track_id,,,..." row
        col = {(g, f): i for i, (g, f) in enumerate(zip(groups, fields))}

        def get(row: list[str], key: tuple[str, str]) -> str | None:
            i = col.get(key)
            v = row[i].strip() if i is not None and i < len(row) else ""
            return v or None

        out: dict[int, FmaTrack] = {}
        for row in reader:
            if not row or not row[0].isdigit():
                continue
            duration = get(row, ("track", "duration"))
            tid = int(row[0])
            out[tid] = FmaTrack(
                track_id=tid,
                title=get(row, ("track", "title")),
                artist=get(row, ("artist", "name")),
                album=get(row, ("album", "title")),
                genre=get(row, ("track", "genre_top")),
                subset=get(row, ("set", "subset")),
                duration_s=float(duration) if duration else None,
            )
        return out
