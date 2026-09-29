import numpy as np
import pytest

from constellation.config import MatchConfig
from constellation.fingerprint import fingerprint
from constellation.matching import confidence_score, join_offsets, match_fingerprint, score_candidates
from constellation.storage import LookupResult

from conftest import SR, add_white_noise, synth_song


def _clip(audio: np.ndarray, start_s: float, dur_s: float) -> np.ndarray:
    s = int(start_s * SR)  # arbitrary sample, NOT aligned to the hop grid
    return audio[s : s + int(dur_s * SR)]


def test_join_offsets_handles_duplicate_hashes():
    q_h = np.array([7, 7, 9])
    q_t = np.array([1, 4, 2])
    hits = LookupResult(np.array([7, 9, 8]), np.array([1, 2, 3]), np.array([10, 20, 30]))
    songs, offsets = join_offsets(q_h, q_t, hits)
    got = sorted(zip(songs.tolist(), offsets.tolist()))
    assert got == [(1, 6), (1, 9), (2, 18)]  # hash 8 is not in the query


def test_score_candidates_tolerance_merges_adjacent_bins():
    songs = np.array([1] * 6 + [2] * 4)
    offsets = np.array([100, 100, 100, 101, 101, 101, 50, 50, 50, 50])
    top = score_candidates(songs, offsets, MatchConfig(offset_tolerance_frames=1))
    assert top[0].song_id == 1 and top[0].aligned_matches == 6
    exact = score_candidates(songs, offsets, MatchConfig(offset_tolerance_frames=0))
    assert exact[0].song_id == 2  # without tolerance the split bin loses


def test_confidence_bounds():
    cfg = MatchConfig(confidence_saturation=20)
    assert confidence_score(0, 0, cfg) == 0.0
    assert confidence_score(40, 0, cfg) == 1.0
    assert confidence_score(40, 40, cfg) == 0.0
    assert 0 < confidence_score(5, 1, cfg) < confidence_score(30, 1, cfg)


@pytest.mark.parametrize("seed,start_s", [(0, 3.3), (7, 12.71), (13, 20.05), (19, 0.0)])
def test_clean_clip_matches_with_correct_offset(library, library_audio, fp_cfg, seed, start_s):
    store, ids = library
    result = match_fingerprint(fingerprint(_clip(library_audio[seed], start_s, 5), fp_cfg), store, fp_cfg)
    assert result.is_match
    assert result.song_id == ids[seed]
    assert abs(result.offset_s - start_s) <= 2 * fp_cfg.frames_to_seconds(1)
    assert result.confidence > 0.5


@pytest.mark.parametrize("snr_db", [15, 5, 0])
def test_noisy_clip_matches(library, library_audio, fp_cfg, snr_db):
    store, ids = library
    correct = 0
    for seed in range(0, 20, 2):
        clip = add_white_noise(_clip(library_audio[seed], 8.0, 10), snr_db, seed=seed)
        result = match_fingerprint(fingerprint(clip, fp_cfg), store, fp_cfg)
        correct += result.is_match and result.song_id == ids[seed]
    assert correct >= 9  # of 10


def test_louder_or_quieter_clip_still_matches(library, library_audio, fp_cfg):
    store, ids = library
    clip = _clip(library_audio[4], 5.0, 5) * 0.03
    assert match_fingerprint(fingerprint(clip, fp_cfg), store, fp_cfg).song_id == ids[4]


def test_unknown_song_is_no_match(library, fp_cfg):
    store, _ = library
    for seed in (100, 101, 102):
        result = match_fingerprint(fingerprint(_clip(synth_song(seed), 5.0, 10), fp_cfg), store, fp_cfg)
        assert not result.is_match and result.song_id is None


def test_pure_noise_is_no_match(library, fp_cfg):
    store, _ = library
    noise = np.random.default_rng(5).standard_normal(SR * 10).astype(np.float32) * 0.3
    assert not match_fingerprint(fingerprint(noise, fp_cfg), store, fp_cfg).is_match


def test_silence_is_no_match(library, fp_cfg):
    store, _ = library
    result = match_fingerprint(fingerprint(np.zeros(SR * 5, np.float32), fp_cfg), store, fp_cfg)
    assert not result.is_match and result.query_hashes == 0
