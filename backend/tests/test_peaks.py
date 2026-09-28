import numpy as np

from constellation.config import FingerprintConfig
from constellation.fingerprint import find_peaks, spectrogram

from conftest import synth_song


def test_finds_planted_peaks():
    cfg = FingerprintConfig(peaks_per_second=100)
    spec = np.full((512, 200), -30.0, np.float32)
    planted = {(10, 50), (60, 300), (150, 100)}  # (time, freq)
    for t, f in planted:
        spec[f, t] = 20.0
    times, freqs = find_peaks(spec, cfg)
    assert set(zip(times.tolist(), freqs.tolist())) == planted


def test_silence_has_no_peaks(fp_cfg):
    times, _ = find_peaks(spectrogram(np.zeros(11_025 * 3, np.float32), fp_cfg), fp_cfg)
    assert times.size == 0


def test_density_cap(fp_cfg):
    audio = synth_song(1, duration_s=10)
    times, _ = find_peaks(spectrogram(audio, fp_cfg), fp_cfg)
    block = times // round(fp_cfg.frames_per_second)
    assert np.bincount(block).max() <= fp_cfg.peaks_per_second
    assert times.size > 5 * fp_cfg.peaks_per_second  # and the cap isn't starving us


def test_volume_invariance(fp_cfg):
    audio = synth_song(2, duration_s=10)
    a = find_peaks(spectrogram(audio, fp_cfg), fp_cfg)
    b = find_peaks(spectrogram(audio * 0.05, fp_cfg), fp_cfg)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_min_freq_filter():
    cfg = FingerprintConfig(min_freq_hz=500)
    audio = synth_song(3, duration_s=10)
    _, freqs = find_peaks(spectrogram(audio, cfg), cfg)
    assert freqs.min() >= 500 * cfg.n_fft / cfg.sample_rate
