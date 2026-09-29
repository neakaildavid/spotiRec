import pickle

import numpy as np
import pytest

from constellation.embedding import HandcraftedEmbedder, l2_normalize, split_windows, standardize
from constellation.similarity import top_k

from conftest import synth_song

SR = 1000  # tiny rate keeps window tests readable


def test_split_windows_full_and_tail():
    x = np.arange(25 * SR, dtype=np.float32)
    wins = split_windows(x, SR, window_s=10, min_last_s=3)
    assert [w.size for w in wins] == [10 * SR, 10 * SR, 5 * SR]
    assert wins[1][0] == 10 * SR


def test_split_windows_drops_short_tail_and_keeps_short_input():
    assert len(split_windows(np.zeros(21 * SR), SR, 10, min_last_s=3)) == 2  # 1 s tail dropped
    only = split_windows(np.zeros(4 * SR), SR, 10)
    assert len(only) == 1 and only[0].size == 4 * SR


def test_split_windows_caps_long_inputs_evenly():
    x = np.arange(300 * SR, dtype=np.float32)  # 30 windows
    wins = split_windows(x, SR, 10, max_windows=5)
    assert len(wins) == 5
    assert wins[0][0] == 0 and wins[-1][0] == 290 * SR  # spans the whole track


def test_l2_normalize():
    v = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(v[0], [0.6, 0.8], rtol=1e-6)
    assert np.all(np.isfinite(v))


def test_top_k_orders_and_excludes():
    rng = np.random.default_rng(0)
    M = l2_normalize(rng.standard_normal((50, 8)))
    q = M[7]
    idx, scores = top_k(q, M, 5)
    assert idx[0] == 7 and scores[0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.diff(scores) <= 0)
    idx2, _ = top_k(q, M, 5, exclude=np.array([7]))
    assert 7 not in idx2 and len(idx2) == 5
    assert len(top_k(q, M, 500)[0]) == 50


def test_handcrafted_features_shape_and_determinism():
    h = HandcraftedEmbedder()
    x = synth_song(3, duration_s=8, sr=h.sample_rate)
    a, b = h.raw_features(x), h.raw_features(x)
    assert a.shape == (h.dim,) and np.array_equal(a, b)


def test_standardize_zero_mean_unit_rows():
    raw = np.random.default_rng(1).normal(50, 10, (100, 6))
    z = standardize(raw)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, rtol=1e-5)


# ---------------------------------------------------------------- CLAP (needs weights)

def _clap_available() -> bool:
    try:
        import torch  # noqa: F401
        from huggingface_hub import try_to_load_from_cache

        from constellation.embedding.clap import MODEL_ID
        return isinstance(try_to_load_from_cache(MODEL_ID, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


clap_only = pytest.mark.skipif(not _clap_available(), reason="CLAP weights not in local HF cache")


@pytest.fixture(scope="module")
def clap():
    from constellation.embedding import ClapEmbedder

    return ClapEmbedder(device="cpu")


@pytest.mark.model
@clap_only
def test_clap_audio_embedding_is_unit_and_deterministic(clap):
    x = synth_song(1, duration_s=25, sr=clap.sample_rate)
    a, b = clap.embed_audio(x), clap.embed_audio(x)
    assert a.shape == (512,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert clap.embed_windows(x).shape == (3, 512)  # 10 s + 10 s + 5 s tail


@pytest.mark.model
@clap_only
def test_clap_clip_is_closest_to_its_own_song(clap):
    songs = {s: synth_song(s, duration_s=20, sr=clap.sample_rate) for s in (10, 11, 12)}
    lib = np.stack([clap.embed_audio(x) for x in songs.values()])
    clip = songs[11][3 * clap.sample_rate : 11 * clap.sample_rate]
    assert int(np.argmax(lib @ clap.embed_audio(clip))) == 1


@pytest.mark.model
@clap_only
def test_clap_text_and_pickle(clap):
    t = clap.embed_text(["solo piano", "heavy metal"])
    assert t.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-4)
    clone = pickle.loads(pickle.dumps(clap))  # workers receive the embedder without the loaded model
    assert clone._model is None and clone.version == clap.version
