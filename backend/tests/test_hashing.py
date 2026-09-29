import numpy as np
import pytest

from constellation.config import FingerprintConfig
from constellation.fingerprint import generate_hashes, pack_hash, unpack_hash


def test_pack_unpack_roundtrip(fp_cfg):
    rng = np.random.default_rng(0)
    f1 = rng.integers(0, 512, 1000)
    f2 = rng.integers(0, 512, 1000)
    dt = rng.integers(0, 256, 1000)
    h = pack_hash(f1, f2, dt, fp_cfg)
    assert h.max() < 2**31  # fits Postgres INTEGER
    for a, b in zip(unpack_hash(h, fp_cfg), (f1, f2, dt)):
        np.testing.assert_array_equal(a, b)


def test_config_rejects_overflowing_layout():
    with pytest.raises(ValueError):
        FingerprintConfig(n_fft=2048)  # 1024 bins don't fit in 9 bits
    with pytest.raises(ValueError):
        FingerprintConfig(target_dt_max=300)


def test_target_zone_and_fan_out():
    cfg = FingerprintConfig(fan_out=3, target_dt_min=1, target_dt_max=10, target_df_max=20)
    times = np.array([0, 2, 3, 4, 5, 30])
    freqs = np.array([100, 105, 200, 110, 115, 100])  # (3, 200) is out of df range
    hashes, anchors = generate_hashes(times, freqs, cfg)
    f1, f2, dt = unpack_hash(hashes, cfg)
    assert np.all((dt >= 1) & (dt <= 10))
    assert np.all(np.abs(f2 - f1) <= 20)
    assert np.bincount(anchors).max() <= 3
    # Anchor at t=0 pairs with the nearest in-zone targets: t=2, 4, 5.
    first = anchors == 0
    assert sorted(dt[first].tolist()) == [2, 4, 5]
    # The peak at t=30 has nothing in its zone and is never an anchor.
    assert 30 not in anchors


def test_time_translation_invariance(fp_cfg):
    rng = np.random.default_rng(1)
    times = np.sort(rng.integers(0, 500, 200))
    freqs = rng.integers(0, 512, 200)
    h1, a1 = generate_hashes(times, freqs, fp_cfg)
    h2, a2 = generate_hashes(times + 1234, freqs, fp_cfg)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(a1 + 1234, a2)


def test_too_few_peaks(fp_cfg):
    h, a = generate_hashes(np.array([5]), np.array([10]), fp_cfg)
    assert h.size == 0 and a.size == 0
