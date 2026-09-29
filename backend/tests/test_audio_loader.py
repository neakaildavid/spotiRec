import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from constellation.audio import AudioDecodeError, load_audio, load_audio_bytes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def stereo_wav(tmp_path):
    sr = 44_100
    t = np.arange(sr * 2) / sr
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, left], axis=1).astype(np.float32)
    path = tmp_path / "tone.wav"
    wavfile.write(path, sr, stereo)
    return path


def test_decodes_to_mono_at_target_rate(stereo_wav):
    x = load_audio(stereo_wav, 11_025)
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(x.size - 2 * 11_025) < 50
    assert 0.65 < np.abs(x).max() < 0.75  # 0.5 * sqrt(2): ffmpeg's power-preserving downmix


def test_seek_and_duration(stereo_wav):
    x = load_audio(stereo_wav, 11_025, start_s=0.5, duration_s=1.0)
    assert abs(x.size - 11_025) < 50


def test_bytes_roundtrip(stereo_wav):
    x = load_audio_bytes(stereo_wav.read_bytes(), 11_025, suffix=".wav")
    assert abs(x.size - 2 * 11_025) < 50


def test_garbage_raises(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(AudioDecodeError):
        load_audio(bad)
