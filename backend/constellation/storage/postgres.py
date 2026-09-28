"""PostgreSQL implementation of :class:`~constellation.storage.base.Storage`."""

from __future__ import annotations

import os
import struct
import threading
from contextlib import contextmanager
from importlib import resources
from typing import Iterator

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from constellation.models import IndexStatus, NewSong, Song
from constellation.storage.base import SONG_SORT_FIELDS, LookupResult

DEFAULT_DSN = "postgresql://constellation:constellation@localhost:5433/constellation"

# Postgres binary COPY framing (see "COPY ... FORMAT binary" in the Postgres docs).
_COPY_HEADER = b"PGCOPY\n\xff\r\n\x00" + struct.pack("!ii", 0, 0)
_COPY_TRAILER = struct.pack("!h", -1)
# Each row: int16 field count, then (int32 length, int32 value) per field.
_FP_ROW = np.dtype(
    [("n", ">i2"), ("l1", ">i4"), ("hash", ">i4"), ("l2", ">i4"), ("song", ">i4"), ("l3", ">i4"), ("t", ">i4")]
)

_SONG_COLS = "id, source, source_id, title, artist, album, duration_s, file_path, content_hash, created_at"


def database_url() -> str:
    return os.environ.get("CONSTELLATION_DATABASE_URL", DEFAULT_DSN)


def schema_sql() -> str:
    return resources.files("constellation.storage").joinpath("schema.sql").read_text()


def _encode_fingerprints_binary(song_id: int, hashes: np.ndarray, anchor_times: np.ndarray) -> bytes:
    """Build a binary COPY payload with numpy instead of a Python loop per row.

    Row-by-row text COPY spends most of its time in Python formatting ints;
    one vectorized big-endian struct array is ~20x faster for bulk ingestion.
    """
    rows = np.empty(hashes.size, dtype=_FP_ROW)
    rows["n"] = 3
    rows["l1"] = rows["l2"] = rows["l3"] = 4
    rows["hash"] = hashes
    rows["song"] = song_id
    rows["t"] = anchor_times
    return _COPY_HEADER + rows.tobytes() + _COPY_TRAILER


class PostgresStorage:
    """Postgres-backed store using a connection pool.

    Connections run in autocommit mode; ``transaction()`` pins one pooled
    connection to the current thread so every storage call made inside the
    ``with`` block joins the same transaction. That works for the CLI (single
    writer thread) and for FastAPI sync endpoints (one request per worker thread).
    """

    def __init__(self, dsn: str | None = None, min_size: int = 1, max_size: int = 10) -> None:
        self.dsn = dsn or database_url()
        self.pool = ConnectionPool(
            self.dsn, min_size=min_size, max_size=max_size, kwargs={"autocommit": True}, open=True
        )
        self._local = threading.local()

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> "PostgresStorage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- connection / transaction plumbing ---
    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            yield conn
            return
        with self.pool.connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:  # nested: savepoint on the pinned connection
            with conn.transaction():
                yield
            return
        with self.pool.connection() as conn:
            self._local.conn = conn
            try:
                with conn.transaction():
                    yield
            finally:
                self._local.conn = None

    def warm_up(self) -> None:
        """Best-effort: wait for the pool, then load the fingerprint table and
        its hash index into Postgres' buffer cache with ``pg_prewarm``.

        Without this, the first identify after a restart reads the B-tree from
        disk page by page (seconds instead of milliseconds). Needs the
        pg_prewarm contrib extension (bundled with the official image) and the
        right to create it; silently skipped otherwise.
        """
        self.pool.wait()
        try:
            with self._conn() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
                conn.execute("SELECT pg_prewarm('fingerprints_hash_idx'), pg_prewarm('fingerprints')")
        except psycopg.Error:
            pass

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(schema_sql())

    # --- SongStore ---
    def add_song(self, song: NewSong) -> Song:
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""INSERT INTO songs (source, source_id, title, artist, album, duration_s, file_path, content_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING {_SONG_COLS}""",
                (song.source, song.source_id, song.title, song.artist, song.album,
                 song.duration_s, song.file_path, song.content_hash),
            )
            return Song(**cur.fetchone())

    def _songs_where(self, where: str, params: tuple, suffix: str = "") -> list[Song]:
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_SONG_COLS} FROM songs {where} {suffix}", params)
            return [Song(**r) for r in cur.fetchall()]

    def get_song(self, song_id: int) -> Song | None:
        rows = self._songs_where("WHERE id = %s", (song_id,))
        return rows[0] if rows else None

    def get_song_by_content_hash(self, content_hash: str) -> Song | None:
        rows = self._songs_where("WHERE content_hash = %s", (content_hash,))
        return rows[0] if rows else None

    def get_songs(self, song_ids: list[int]) -> dict[int, Song]:
        return {s.id: s for s in self._songs_where("WHERE id = ANY(%s)", (list(song_ids),))}

    @staticmethod
    def _search_clause(query: str | None) -> tuple[str, tuple]:
        """Substring search on title/artist. A sequential scan is fine at ~10^4
        songs; at larger scale this would get a pg_trgm GIN index."""
        if not query:
            return "", ()
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        return "WHERE title ILIKE %s OR artist ILIKE %s", (pattern, pattern)

    def list_songs(
        self,
        limit: int = 100,
        offset: int = 0,
        query: str | None = None,
        sort: str = "id",
        descending: bool = False,
    ) -> list[Song]:
        if sort not in SONG_SORT_FIELDS:  # whitelist: the column name is interpolated
            raise ValueError(f"cannot sort by {sort!r}")
        where, params = self._search_clause(query)
        col = f"lower({sort})" if sort in ("title", "artist") else sort
        direction = "DESC" if descending else "ASC"
        suffix = f"ORDER BY {col} {direction} NULLS LAST, id LIMIT %s OFFSET %s"
        return self._songs_where(where, (*params, limit, offset), suffix)

    def count_songs(self, query: str | None = None) -> int:
        where, params = self._search_clause(query)
        with self._conn() as conn:
            return conn.execute(f"SELECT count(*) FROM songs {where}", params).fetchone()[0]

    def random_song(self) -> Song | None:
        # ORDER BY random() scans the table; fine for ~10^4 songs.
        rows = self._songs_where("", (), "ORDER BY random() LIMIT 1")
        return rows[0] if rows else None

    # --- FingerprintStore ---
    def add_fingerprints(self, song_id: int, hashes: np.ndarray, anchor_times: np.ndarray) -> None:
        if hashes.size == 0:
            return
        payload = _encode_fingerprints_binary(song_id, hashes, anchor_times)
        with self._conn() as conn, conn.cursor() as cur:
            with cur.copy("COPY fingerprints (hash, song_id, anchor_time) FROM STDIN (FORMAT binary)") as copy:
                copy.write(payload)

    def delete_fingerprints(self, song_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM fingerprints WHERE song_id = %s", (song_id,))

    def lookup(self, hashes: np.ndarray) -> LookupResult:
        uniq = np.unique(np.asarray(hashes, np.int64))
        if uniq.size == 0:
            return LookupResult.empty()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT hash, song_id, anchor_time FROM fingerprints WHERE hash = ANY(%s)",
                (uniq.tolist(),),
            ).fetchall()
        if not rows:
            return LookupResult.empty()
        arr = np.asarray(rows, dtype=np.int64)
        return LookupResult(arr[:, 0], arr[:, 1], arr[:, 2])

    # --- IndexStatusStore ---
    def get_index_status(self, song_id: int, index_name: str) -> IndexStatus | None:
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT song_id, index_name, version, item_count, indexed_at FROM song_indexes "
                "WHERE song_id = %s AND index_name = %s",
                (song_id, index_name),
            )
            row = cur.fetchone()
            return IndexStatus(**row) if row else None

    def set_index_status(self, status: IndexStatus) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO song_indexes (song_id, index_name, version, item_count)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (song_id, index_name)
                   DO UPDATE SET version = EXCLUDED.version, item_count = EXCLUDED.item_count,
                                 indexed_at = now()""",
                (status.song_id, status.index_name, status.version, status.item_count),
            )

    # --- stats (used by the CLI and eval) ---
    def stats(self) -> dict[str, int]:
        """Row counts and on-disk sizes (bytes, including indexes)."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT (SELECT count(*) FROM songs),
                          (SELECT count(*) FROM fingerprints),
                          pg_total_relation_size('fingerprints'),
                          pg_relation_size('fingerprints'),
                          pg_indexes_size('fingerprints'),
                          pg_total_relation_size('songs') + pg_total_relation_size('song_indexes'),
                          pg_database_size(current_database())"""
            ).fetchone()
        keys = ("songs", "fingerprints", "fingerprints_total_bytes", "fingerprints_table_bytes",
                "fingerprints_index_bytes", "metadata_bytes", "database_bytes")
        return dict(zip(keys, row))
