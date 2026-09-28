# Constellation

Shazam-style music identification (Phase 1) and audio-similarity discovery (Phase 2, planned).
Record a few seconds of a song, and Constellation tells you what it is and where in the song you are.

> Status: **Milestone 1** (fingerprinting and matching core, pure Python, tested) is done.
> Still to come: Postgres storage + ingestion CLI, evaluation, FastAPI, React frontend.

## How it works

```
audio ──ffmpeg──▶ mono 11,025 Hz ──STFT──▶ log spectrogram ──max filter──▶ constellation map
                                                                               │
                              (f1, f2, Δt) packed into a 26-bit int ◀──pair anchors with target zone
                                                                               │
          index: hash → (song_id, anchor_time)      query: look up hashes, histogram
                                                    (db_time − query_time) per song,
                                                    tallest bin wins → song + offset
```

| Step | Module | Key idea |
|---|---|---|
| Decode | `audio/loader.py` | ffmpeg decodes any format (incl. browser webm/opus), downmixes and resamples in one native pass |
| Spectrogram | `fingerprint/spectrogram.py` | Hann-windowed STFT in dB; frames are not centered, so frame indices map to absolute sample positions |
| Peaks | `fingerprint/peaks.py` | 2D max filter + threshold relative to the clip's loudest point (volume invariant) + top-k per second (bounded density) |
| Hashing | `fingerprint/hashing.py` | Peak *pairs* carry about 26 bits (vs. about 9 for one peak) and store only Δt, so they're invariant to where the clip starts |
| Matching | `matching/matcher.py` | True hits all agree on one offset; chance hits scatter. The tallest offset-histogram bin measures time-coherent agreement |
| Storage | `storage/` | `Protocol` interfaces; in-memory implementation now, Postgres next |

All parameters live in `config.py` (`FingerprintConfig`, `MatchConfig`). `FingerprintConfig.version()` hashes them, so an index knows which config produced it.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest            # requires ffmpeg on PATH for the loader tests
```
