import hashlib
import io
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from constellation.api.main import create_app
from constellation.api.settings import Settings
from constellation.fingerprint import fingerprint
from constellation.models import NewSong
from constellation.embedding.base import l2_normalize
from constellation.pipeline import FingerprintIndexer
from constellation.storage import MemoryStorage

from conftest import SR, add_white_noise, synth_song

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, SR, x.astype(np.float32))
    return buf.getvalue()


class FakeEmbedder:
    """Deterministic CLAP stand-in: band energies for audio, keyword vectors for text."""

    name = "fake"
    dim = 16
    sample_rate = SR
    version = "fake-v1"

    def __init__(self):
        self.text_calls = 0

    def embed_audio(self, samples):
        spec = np.abs(np.fft.rfft(samples[: 2**16]))
        return l2_normalize(np.array([b.mean() for b in np.array_split(spec, self.dim)], np.float32))

    def embed_text(self, texts):
        self.text_calls += 1
        rows = [np.random.default_rng(abs(hash(t)) % 2**32).random(self.dim) for t in texts]
        return l2_normalize(np.stack(rows).astype(np.float32))


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def client(tmp_path, embedder):
    """App over an in-memory library of 5 synthetic songs, each backed by a real wav file.

    Artists alternate (Artist 0, Artist 1, Artist 0, ...) and genres by seed.
    """
    store = MemoryStorage()
    for seed in range(5):
        audio = synth_song(seed, duration_s=20)
        path = tmp_path / f"lib{seed}.wav"
        wavfile.write(path, SR, audio)
        song = store.add_song(NewSong(
            title=f"Song {seed}", artist=f"Artist {seed % 2}", album=None, duration_s=20.0,
            file_path=str(path), content_hash=hashlib.sha256(path.read_bytes()).hexdigest(), source="fma",
            genre="Rock" if seed < 3 else "Folk",
        ))
        fp = fingerprint(audio)
        store.add_fingerprints(song.id, fp.hashes, fp.anchor_times)
        store.add_embedding(song.id, embedder.name, embedder.embed_audio(audio))
    app = create_app(storage=store, settings=Settings(upload_dir=tmp_path / "uploads"),
                     indexers=[FingerprintIndexer()], embedder=embedder)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["songs"] == 5 and body["fingerprint_version"]
    assert body["embedding_model"] == "fake"


def test_identify_match(client):
    clip = add_white_noise(synth_song(3, duration_s=20)[int(7.4 * SR) : int(14.4 * SR)], 10)
    r = client.post("/identify", files={"file": ("clip.wav", wav_bytes(clip), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["song"]["title"] == "Song 3"
    assert abs(body["offset_s"] - 7.4) < 0.05
    assert 0 < body["confidence"] <= 1
    assert body["query"]["peaks"] and body["timing"]["total_ms"] > 0
    assert "file_path" not in body["song"]


def test_identify_no_match_is_200(client):
    clip = synth_song(999, duration_s=6)
    r = client.post("/identify", files={"file": ("clip.wav", wav_bytes(clip), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["match"] is False and r.json()["song"] is None


def test_identify_rejects_garbage_and_short_clips(client):
    r = client.post("/identify", files={"file": ("x.webm", b"\x1a\x45\xdf\xa3garbage", "audio/webm")})
    assert r.status_code == 422
    short = client.post("/identify", files={"file": ("s.wav", wav_bytes(synth_song(1, 0.5)), "audio/wav")})
    assert short.status_code == 422 and "too short" in short.json()["detail"]
    assert client.post("/identify").status_code == 422  # no file


def test_identify_upload_size_limit(tmp_path):
    app = create_app(storage=MemoryStorage(), settings=Settings(upload_dir=tmp_path, max_identify_bytes=1000), embedder=None)
    with TestClient(app) as c:
        r = c.post("/identify", files={"file": ("big.wav", b"\x00" * 5000, "audio/wav")})
    assert r.status_code == 413


def test_list_search_sort_paginate(client):
    page = client.get("/songs", params={"limit": 2}).json()
    assert page["total"] == 5 and len(page["items"]) == 2 and page["items"][0]["id"] == 1
    assert client.get("/songs", params={"q": "artist 1"}).json()["total"] == 2
    desc = client.get("/songs", params={"sort": "title", "order": "desc"}).json()["items"]
    assert [s["title"] for s in desc] == [f"Song {i}" for i in range(4, -1, -1)]
    assert client.get("/songs", params={"sort": "file_path"}).status_code == 422  # not whitelisted
    assert client.get("/songs", params={"limit": 10_000}).status_code == 422


def test_get_and_random(client):
    assert client.get("/songs/2").json()["title"] == "Song 1"
    assert client.get("/songs/999").status_code == 404
    assert client.get("/songs/random").json()["id"] in range(1, 6)


def test_audio_supports_range_requests(client):
    full = client.get("/songs/1/audio")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    part = client.get("/songs/1/audio", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.content == full.content[100:200]


def test_add_song_then_identify_then_dedupe(client):
    audio = synth_song(42, duration_s=15)
    data = wav_bytes(audio)
    r = client.post("/songs", files={"file": ("my track.wav", data, "audio/wav")}, data={"artist": "Me"})
    assert r.status_code == 201
    body = r.json()
    assert body["outcome"] == "ingested" and body["indexes"] == ["fingerprint"]
    assert body["song"]["title"] == "my track" and body["song"]["artist"] == "Me"

    clip = wav_bytes(audio[3 * SR : 9 * SR])
    found = client.post("/identify", files={"file": ("c.wav", clip, "audio/wav")}).json()
    assert found["match"] and found["song"]["id"] == body["song"]["id"]

    again = client.post("/songs", files={"file": ("renamed.wav", data, "audio/wav")})
    assert again.status_code == 200 and again.json()["outcome"] == "skipped"
    assert client.get("/health").json()["songs"] == 6


def test_add_song_rejects_undecodable(client, tmp_path):
    r = client.post("/songs", files={"file": ("bad.mp3", b"nope" * 100, "audio/mpeg")})
    assert r.status_code == 422
    assert not any((tmp_path / "uploads").glob("*.mp3"))  # stored file cleaned up


# ---------------------------------------------------------------- Phase 2: discovery

def test_similar_songs_excludes_self_sorted_and_caps_artist(client):
    r = client.get("/songs/1/similar", params={"k": 4})
    assert r.status_code == 200
    body = r.json()
    ids = [i["song"]["id"] for i in body["items"]]
    scores = [i["score"] for i in body["items"]]
    assert 1 not in ids and scores == sorted(scores, reverse=True)
    # Song 1 is by Artist 0, which counts toward the cap of 2: at most one more Artist 0 song.
    assert sum(i["song"]["artist"] == "Artist 0" for i in body["items"]) <= 1
    assert len(ids) == 3  # 2 by Artist 1 + 1 by Artist 0
    assert body["model"] == "fake" and body["embed_ms"] == 0


def test_similar_errors(client):
    assert client.get("/songs/999/similar").status_code == 404
    audio = synth_song(77, duration_s=6)
    added = client.post("/songs", files={"file": ("new.wav", wav_bytes(audio), "audio/wav")}).json()
    r = client.get(f"/songs/{added['song']['id']}/similar")  # fingerprint-only indexers in this app
    assert r.status_code == 409 and "embedding" in r.json()["detail"]
    assert client.get("/songs/1/similar", params={"k": 0}).status_code == 422


def test_discover_by_audio_finds_source_song(client):
    # Tests the wiring (upload -> decode -> embed -> search); retrieval quality is
    # measured by eval/run_similarity_eval.py. The fake embedder only looks at the
    # first ~6 s, so the clip starts where the library vector was taken.
    clip = synth_song(3, duration_s=20)[: 10 * SR]
    r = client.post("/discover/audio", files={"file": ("c.wav", wav_bytes(clip), "audio/wav")}, params={"k": 3})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["song"]["title"] == "Song 3" and len(items) == 3
    assert r.json()["embed_ms"] >= 0


def test_discover_text_and_cache(client, embedder):
    r1 = client.post("/discover/text", json={"query": "Mellow  acoustic guitar", "k": 3})
    r2 = client.post("/discover/text", json={"query": "mellow acoustic guitar", "k": 3})
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["items"] == r2.json()["items"] and len(r1.json()["items"]) == 3
    assert embedder.text_calls == 1  # normalized query hit the LRU cache
    assert client.post("/discover/text", json={"query": ""}).status_code == 422
    assert client.post("/discover/text", json={"query": "x" * 201}).status_code == 422


def test_discovery_without_model(tmp_path):
    store = MemoryStorage()
    app = create_app(storage=store, settings=Settings(upload_dir=tmp_path), embedder=None,
                     indexers=[FingerprintIndexer()])
    with TestClient(app) as c:
        assert c.get("/health").json()["embedding_model"] is None
        assert c.post("/discover/text", json={"query": "piano"}).status_code == 503
        clip = wav_bytes(synth_song(1, duration_s=4))
        assert c.post("/discover/audio", files={"file": ("c.wav", clip, "audio/wav")}).status_code == 503


def test_genres_and_genre_filter(client):
    assert client.get("/genres").json() == [{"genre": "Rock", "songs": 3}, {"genre": "Folk", "songs": 2}]
    page = client.get("/songs", params={"genre": "Folk"}).json()
    assert page["total"] == 2 and all(s["genre"] == "Folk" for s in page["items"])
