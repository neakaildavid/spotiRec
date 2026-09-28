import shutil
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
from scipy.io import wavfile

from constellation.config import FingerprintConfig
from constellation.fingerprint import fingerprint
from constellation.matching import match_fingerprint
from constellation.pipeline import FingerprintIndexer, Outcome, SongSource, ingest_many, process_song
from constellation.storage import MemoryStorage

from conftest import SR, synth_song

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def song_files(tmp_path):
    paths = []
    for seed in range(4):
        p = tmp_path / f"song{seed}.wav"
        wavfile.write(p, SR, synth_song(seed, duration_s=12))
        paths.append(p)
    return paths


@dataclass(frozen=True)
class FakeEmbeddingIndexer:
    """Stands in for Phase 2: a second index at a different sample rate."""

    name: ClassVar[str] = "embedding"
    sample_rate: int = 16_000
    version: str = "v1"

    def compute(self, samples):
        return float(np.sqrt(np.mean(samples**2)))

    def write(self, storage, song_id, payload):
        storage.embeddings = getattr(storage, "embeddings", {})
        storage.embeddings[song_id] = payload
        return 1

    def delete(self, storage, song_id):
        getattr(storage, "embeddings", {}).pop(song_id, None)


def test_ingest_then_identify(song_files):
    store = MemoryStorage()
    counts = ingest_many(store, [SongSource(p) for p in song_files], [FingerprintIndexer()])
    assert counts[Outcome.INGESTED] == 4
    song = store.get_song(2)
    assert song.title == "song1" and abs(song.duration_s - 12) < 0.1
    assert store.get_index_status(song.id, "fingerprint").item_count > 0

    clip = synth_song(1, duration_s=12)[SR * 4 : SR * 9]
    assert match_fingerprint(fingerprint(clip), store).song_id == song.id


def test_rerun_is_idempotent(song_files):
    store = MemoryStorage()
    sources = [SongSource(p) for p in song_files]
    ingest_many(store, sources, [FingerprintIndexer()])
    n_rows = sum(len(v) for v in store._index.values())
    counts = ingest_many(store, sources, [FingerprintIndexer()])
    assert counts[Outcome.SKIPPED] == 4 and counts[Outcome.INGESTED] == 0
    assert sum(len(v) for v in store._index.values()) == n_rows


def test_renamed_duplicate_is_not_reingested(song_files, tmp_path):
    store = MemoryStorage()
    process_song(store, SongSource(song_files[0]), [FingerprintIndexer()])
    copy = tmp_path / "renamed.wav"
    copy.write_bytes(song_files[0].read_bytes())
    assert process_song(store, SongSource(copy), [FingerprintIndexer()]).outcome is Outcome.SKIPPED
    assert store.count_songs() == 1


def test_config_change_reindexes_only_stale_index(song_files):
    store = MemoryStorage()
    sources = [SongSource(p) for p in song_files[:2]]
    ingest_many(store, sources, [FingerprintIndexer(), FakeEmbeddingIndexer()])
    old_count = store.get_index_status(1, "fingerprint").item_count

    new_fp = FingerprintIndexer(FingerprintConfig(fan_out=2))
    results = []
    ingest_many(store, sources, [new_fp, FakeEmbeddingIndexer()], on_result=results.append)
    assert all(r.outcome is Outcome.UPDATED and r.indexes == ["fingerprint"] for r in results)
    status = store.get_index_status(1, "fingerprint")
    assert status.version == new_fp.version and status.item_count < old_count
    # Old rows were replaced, not appended to.
    assert sum(len(v) for v in store._index.values()) == sum(
        store.get_index_status(i, "fingerprint").item_count for i in (1, 2)
    )


def test_adding_a_new_indexer_backfills_existing_songs(song_files):
    """The Phase 2 path: existing songs get only the new index."""
    store = MemoryStorage()
    sources = [SongSource(p) for p in song_files]
    ingest_many(store, sources, [FingerprintIndexer()])
    results = []
    ingest_many(store, sources, [FingerprintIndexer(), FakeEmbeddingIndexer()], on_result=results.append)
    assert [r.indexes for r in results] == [["embedding"]] * 4
    assert len(store.embeddings) == 4


def test_bad_file_fails_without_stopping_batch(song_files, tmp_path):
    bad = tmp_path / "corrupt.mp3"
    bad.write_bytes(b"\x00" * 1000)
    store = MemoryStorage()
    results = []
    counts = ingest_many(
        store, [SongSource(bad)] + [SongSource(p) for p in song_files], [FingerprintIndexer()],
        on_result=results.append,
    )
    assert counts[Outcome.FAILED] == 1 and counts[Outcome.INGESTED] == 4
    assert store.get_song_by_content_hash(__import__("hashlib").sha256(bad.read_bytes()).hexdigest()) is None


def test_parallel_workers_match_serial(song_files):
    serial, parallel = MemoryStorage(), MemoryStorage()
    sources = [SongSource(p) for p in song_files]
    ingest_many(serial, sources, [FingerprintIndexer()], workers=1)
    ingest_many(parallel, sources, [FingerprintIndexer()], workers=2)
    by_hash = lambda s: {x.content_hash: s.get_index_status(x.id, "fingerprint").item_count for x in s.list_songs()}
    assert by_hash(serial) == by_hash(parallel)
