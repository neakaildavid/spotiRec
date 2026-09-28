import numpy as np

from constellation.config import FingerprintConfig
from constellation.fingerprint import spectrogram

SR = 11_025


def test_shape_and_frame_count(fp_cfg):
    x = np.zeros(SR * 2, np.float32)
    spec = spectrogram(x, fp_cfg)
    assert spec.shape[0] == fp_cfg.n_fft // 2
    assert spec.shape[1] == 1 + (x.size - fp_cfg.n_fft) // fp_cfg.hop_length


def test_pure_tone_peaks_at_right_bin(fp_cfg):
    freq = 1000.0
    t = np.arange(SR) / SR
    spec = spectrogram(np.sin(2 * np.pi * freq * t).astype(np.float32), fp_cfg)
    expected_bin = round(freq * fp_cfg.n_fft / SR)
    assert np.all(np.abs(spec.argmax(axis=0) - expected_bin) <= 1)


def test_short_input_is_padded(fp_cfg):
    assert spectrogram(np.ones(10, np.float32), fp_cfg).shape[1] == 1


def test_configurable_window(fp_cfg):
    cfg = FingerprintConfig(n_fft=512, hop_length=128)
    spec = spectrogram(np.zeros(SR, np.float32), cfg)
    assert spec.shape[0] == 256
