"""Integration tests against the docker-compose Postgres (skipped if unreachable).

Uses a separate ``constellation_test`` database so tests never touch real data.
"""

import os

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from constellation.models import IndexStatus, NewSong  # noqa: E402

pytestmark = pytest.mark.integration

ADMIN_DSN = os.environ.get(
    "CONSTELLATION_TEST_ADMIN_URL", "postgresql://constellation:constellation@localhost:5433/postgres"
)
TEST_DSN = ADMIN_DSN.rsplit("/", 1)[0] + "/constellation_test"


@pytest.fixture(scope="module")
def pg():
    try:
        with psycopg.connect(ADMIN_DSN, autocommit=True, connect_timeout=2) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'constellation_test'").fetchone()
            if not exists:
                conn.execute("CREATE DATABASE constellation_test")
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable (run `docker compose up -d`)")
    from constellation.storage.postgres import PostgresStorage

    store = PostgresStorage(TEST_DSN)
    with store._conn() as conn:
        conn.execute("DROP TABLE IF EXISTS fingerprints, song_indexes, songs CASCADE")
    store.init_schema()
    store.init_schema()  # idempotent
    yield store
    store.close()


def _song(n: int) -> NewSong:
    return NewSong(title=f"t{n}", artist="a", album=None, duration_s=30.0, file_path=f"/x/{n}.mp3",
                   content_hash=f"hash{n}", source="fma", source_id=str(n))


def test_song_crud(pg):
    s = pg.add_song(_song(1))
    assert s.id > 0 and s.created_at is not None
    assert pg.get_song(s.id) == s
    assert pg.get_song_by_content_hash("hash1") == s
    assert pg.get_songs([s.id, 99999]) == {s.id: s}
    assert s in pg.list_songs()
    assert pg.random_song() is not None
    with pytest.raises(psycopg.errors.UniqueViolation):
        pg.add_song(_song(1))


def test_fingerprint_roundtrip_binary_copy(pg):
    s = pg.add_song(_song(2))
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**26, 5000)
    times = rng.integers(0, 1300, 5000)
    pg.add_fingerprints(s.id, hashes, times)

    hit = pg.lookup(hashes[:100])
    assert set(hit.song_ids.tolist()) == {s.id}
    got = set(zip(hit.hashes.tolist(), hit.anchor_times.tolist()))
    assert set(zip(hashes[:100].tolist(), times[:100].tolist())) <= got
    assert len(pg.lookup(np.array([2**27]))) == 0

    pg.delete_fingerprints(s.id)
    assert len(pg.lookup(hashes[:100])) == 0


def test_transaction_rolls_back_everything(pg):
    with pytest.raises(RuntimeError):
        with pg.transaction():
            s = pg.add_song(_song(3))
            pg.add_fingerprints(s.id, np.array([123456]), np.array([7]))
            pg.set_index_status(IndexStatus(s.id, "fingerprint", "v", 1))
            raise RuntimeError("crash mid-song")
    assert pg.get_song_by_content_hash("hash3") is None
    assert len(pg.lookup(np.array([123456]))) == 0


def test_index_status_upsert(pg):
    s = pg.add_song(_song(4))
    pg.set_index_status(IndexStatus(s.id, "fingerprint", "v1", 10))
    pg.set_index_status(IndexStatus(s.id, "fingerprint", "v2", 20))
    st = pg.get_index_status(s.id, "fingerprint")
    assert (st.version, st.item_count) == ("v2", 20)
    assert pg.get_index_status(s.id, "embedding") is None


def test_stats(pg):
    st = pg.stats()
    assert st["songs"] >= 1 and st["fingerprints_total_bytes"] > 0
