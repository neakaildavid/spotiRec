"""Command-line interface.

    constellation db init                      # create tables (idempotent)
    constellation db stats                     # row counts and on-disk sizes
    constellation ingest data/fma_small --metadata data/fma_metadata/tracks.csv --workers 8
    constellation ingest data/fma_small --indexes embedding   # just (back)fill CLAP vectors
    constellation db backfill-genres --metadata data/fma_metadata/tracks.csv
    constellation identify clip.m4a
    constellation serve --reload               # FastAPI on http://localhost:8000 (docs at /docs)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _storage():
    """Open a pooled Postgres store; use as a context manager so the pool closes."""
    from constellation.storage.postgres import PostgresStorage

    return PostgresStorage()


def cmd_db_init(args: argparse.Namespace) -> None:
    with _storage() as s:
        s.init_schema()
        print(f"schema ready at {s.dsn.rsplit('@', 1)[-1]}")


def cmd_db_stats(args: argparse.Namespace) -> None:
    with _storage() as s:
        st = s.stats()
    print(f"songs:              {st['songs']:,}")
    print(f"fingerprints:       {st['fingerprints']:,}")
    print(f"  table:            {_human(st['fingerprints_table_bytes'])}")
    print(f"  indexes:          {_human(st['fingerprints_index_bytes'])}")
    print(f"metadata tables:    {_human(st['metadata_bytes'])}")
    print(f"embeddings:         {st['embeddings']:,}")
    print(f"  table:            {_human(st['embeddings_table_bytes'])}")
    print(f"  HNSW index:       {_human(st['embeddings_index_bytes'])}")
    print(f"database total:     {_human(st['database_bytes'])}")


def cmd_db_backfill_genres(args: argparse.Namespace) -> None:
    """Fill songs.genre from FMA metadata for rows ingested before the column existed."""
    from constellation.metadata.fma import load_tracks

    meta = load_tracks(args.metadata)
    with _storage() as s:
        s.init_schema()
        genres = {}
        offset = 0
        while page := s.list_songs(limit=1000, offset=offset):
            for song in page:
                if song.source == "fma" and song.source_id and song.source_id.isdigit():
                    t = meta.get(int(song.source_id))
                    if t and t.genre:
                        genres[song.id] = t.genre
            offset += len(page)
        n = s.set_genres(genres)
        print(f"updated {n} songs")
        for g, c in s.list_genres():
            print(f"  {g:<14} {c:>5}")


def _discover(root: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())
    return files[:limit] if limit else files


def cmd_ingest(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from constellation.metadata.fma import load_tracks, track_id_from_path
    from constellation.pipeline import Outcome, SongSource, default_indexers, ingest_many
    from constellation.pipeline.indexers import embeddings_available

    files = _discover(Path(args.folder), args.limit)
    if not files:
        sys.exit(f"no audio files under {args.folder}")

    meta = load_tracks(args.metadata) if args.metadata else {}
    sources = []
    for f in files:
        tid = track_id_from_path(f)
        t = meta.get(tid) if tid is not None else None
        sources.append(
            SongSource(
                path=f,
                title=t.title if t else None,
                artist=t.artist if t else None,
                album=t.album if t else None,
                source="fma" if tid is not None else "file",
                source_id=str(tid) if tid is not None else None,
                genre=t.genre if t else None,
            )
        )

    wanted = {x.strip() for x in args.indexes.split(",")} if args.indexes else None
    if wanted and "embedding" in wanted and not embeddings_available():
        sys.exit("embedding index needs the optional extra: pip install -e '.[embeddings]'")
    indexers = [ix for ix in default_indexers() if wanted is None or ix.name.split(":")[0] in wanted]
    if not indexers:
        sys.exit(f"no indexers match --indexes {args.indexes!r} (choose from: fingerprint, embedding)")

    # Two passes: CPU-bound indexers fan out across worker processes; model-based
    # ones run in this process (one model copy, on the GPU if there is one). Each
    # pass is idempotent on its own, so the second only does what's missing.
    groups = [[ix for ix in indexers if ix.parallel], [ix for ix in indexers if not ix.parallel]]
    with _storage() as storage:
        storage.init_schema()
        for group in (g for g in groups if g):
            names = ", ".join(ix.name for ix in group)
            started = time.perf_counter()
            bar = tqdm(total=len(sources), unit="song", dynamic_ncols=True, desc=names)

            def on_result(r) -> None:
                bar.update(1)
                if r.outcome is Outcome.FAILED:
                    bar.write(f"  failed: {r.source.path.name}: {r.error}")

            workers = args.workers if group[0].parallel else 1
            counts = ingest_many(storage, sources, group, workers=workers, on_result=on_result)
            bar.close()
            elapsed = time.perf_counter() - started
            print(f"[{names}] " + " | ".join(f"{o.value}: {n}" for o, n in counts.items()) + f"  ({elapsed:.1f}s)")
            if counts[Outcome.FAILED]:
                print("  failed files are not recorded and will be retried on the next run")


def cmd_identify(args: argparse.Namespace) -> None:
    from constellation.audio import load_audio
    from constellation.config import FingerprintConfig
    from constellation.fingerprint import fingerprint
    from constellation.matching import match_fingerprint

    cfg = FingerprintConfig()
    with _storage() as storage:
        t0 = time.perf_counter()
        fp = fingerprint(load_audio(args.file, cfg.sample_rate), cfg)
        r = match_fingerprint(fp, storage, cfg)
        ms = (time.perf_counter() - t0) * 1000
        song = storage.get_song(r.song_id) if r.is_match else None
    if song is None:
        print(f"no match  (best {r.aligned_matches} aligned, confidence {r.confidence:.2f}, {ms:.0f} ms)")
        return
    m, s = divmod(max(0.0, r.offset_s), 60)
    print(f"{song.title} - {song.artist}")
    print(f"  matched at {int(m)}:{s:04.1f}, confidence {r.confidence:.2f} "
          f"({r.aligned_matches} aligned vs {r.runner_up_matches} runner-up), {ms:.0f} ms")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("constellation.api.main:app", host=args.host, port=args.port, reload=args.reload)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="constellation")
    sub = p.add_subparsers(required=True)

    db = sub.add_parser("db", help="database management").add_subparsers(required=True)
    db.add_parser("init", help="create tables").set_defaults(func=cmd_db_init)
    db.add_parser("stats", help="row counts and sizes").set_defaults(func=cmd_db_stats)
    bf = db.add_parser("backfill-genres", help="fill songs.genre from FMA tracks.csv")
    bf.add_argument("--metadata", required=True)
    bf.set_defaults(func=cmd_db_backfill_genres)

    ing = sub.add_parser("ingest", help="fingerprint a folder of audio files")
    ing.add_argument("folder")
    ing.add_argument("--metadata", help="FMA tracks.csv for titles/artists")
    ing.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ing.add_argument("--limit", type=int, help="only the first N files (sorted by path)")
    ing.add_argument("--indexes", help="comma-separated subset: fingerprint,embedding (default: all available)")
    ing.set_defaults(func=cmd_ingest)

    idf = sub.add_parser("identify", help="identify an audio clip")
    idf.add_argument("file")
    idf.set_defaults(func=cmd_identify)

    srv = sub.add_parser("serve", help="run the HTTP API")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true")
    srv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
