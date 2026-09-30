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
        conn.execute("DROP TABLE IF EXISTS song_embeddings, fingerprints, song_indexes, songs CASCADE")
    store.init_schema()
    store.init_schema()  # idempotent
    yield store
    store.close()


def _song(n: int, genre: str | None = None) -> NewSong:
    return NewSong(title=f"t{n}", artist="a", album=None, duration_s=30.0, file_path=f"/x/{n}.mp3",
                   content_hash=f"hash{n}", source="fma", source_id=str(n), genre=genre)


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


def test_genre_column_filter_and_backfill(pg):
    a = pg.add_song(_song(10, genre="Rock"))
    b = pg.add_song(_song(11))
    assert pg.get_song(a.id).genre == "Rock" and pg.get_song(b.id).genre is None
    assert pg.set_genres({b.id: "Folk", a.id: "Rock"}) == 1  # unchanged rows aren't rewritten
    assert pg.get_song(b.id).genre == "Folk"
    assert {s.id for s in pg.list_songs(genre="Folk")} == {b.id}
    assert pg.count_songs(genre="Rock") == 1
    assert ("Folk", 1) in pg.list_genres()


def _unit(rng, n, d=512):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_embeddings_roundtrip_and_nearest(pg):
    rng = np.random.default_rng(0)
    songs = [pg.add_song(_song(100 + i)) for i in range(30)]
    vecs = _unit(rng, 30)
    for s, v in zip(songs, vecs):
        pg.add_embedding(s.id, "clap-htsat", v)
    np.testing.assert_allclose(pg.get_embedding(songs[3].id, "clap-htsat"), vecs[3], atol=1e-6)
    ids, mat = pg.all_embeddings("clap-htsat")
    assert len(ids) == 30 and mat.shape == (30, 512)

    # A query close to song 5 finds song 5 first; excluding it, never returns it.
    q = vecs[5] + 0.05 * _unit(rng, 1)[0]
    q /= np.linalg.norm(q)
    hits = pg.nearest_embeddings(q, "clap-htsat", 5)
    assert hits[0][0] == songs[5].id and 0.9 < hits[0][1] <= 1.0
    assert all(h[1] >= hits[i + 1][1] for i, h in enumerate(hits[:-1]))
    excl = pg.nearest_embeddings(q, "clap-htsat", 5, exclude=(songs[5].id,))
    assert len(excl) == 5 and songs[5].id not in [h[0] for h in excl]

    pg.add_embedding(songs[3].id, "clap-htsat", vecs[4])  # upsert
    np.testing.assert_allclose(pg.get_embedding(songs[3].id, "clap-htsat"), vecs[4], atol=1e-6)
    pg.delete_embedding(songs[3].id, "clap-htsat")
    assert pg.get_embedding(songs[3].id, "clap-htsat") is None


def test_nearest_query_uses_hnsw_index(pg):
    """The cast and WHERE clause must match the partial expression index."""
    q = "[" + ",".join(["0.04"] * 512) + "]"
    with pg._conn() as conn, conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")  # tiny table: force the planner to consider indexes
        plan = "\n".join(r[0] for r in conn.execute(
            """EXPLAIN SELECT song_id FROM song_embeddings WHERE model = 'clap-htsat'
               ORDER BY vec::vector(512) <=> %s::vector(512) LIMIT 5""", (q,)).fetchall())
    assert "song_embeddings_clap_htsat_hnsw" in plan


def test_nearest_returns_k_even_above_ef_search(pg):
    """HNSW returns at most ef_search rows; the store must raise it for large k."""
    rng = np.random.default_rng(3)
    for i, v in enumerate(_unit(rng, 60)):
        pg.add_embedding(pg.add_song(_song(500 + i)).id, "clap-htsat", v)
    pg.ef_search = 10
    try:
        assert len(pg.nearest_embeddings(_unit(rng, 1)[0], "clap-htsat", 25)) == 25
    finally:
        pg.ef_search = 40
