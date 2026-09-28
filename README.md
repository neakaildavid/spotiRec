# Constellation

Shazam-style music identification (Phase 1) and audio-similarity discovery (Phase 2, planned).
Record a few seconds of a song, and Constellation tells you what it is and where in the song you are.

> Status: Milestones 1–2 are done (fingerprinting core, Postgres storage, ingestion CLI).
> Still to come: evaluation, FastAPI, React frontend.

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
| Storage | `storage/` | `Protocol` interfaces with in-memory and Postgres implementations; swap in e.g. Redis by implementing `add_fingerprints` + `lookup` |
| Ingestion | `pipeline/` | One place that processes a song into **every** index (see below) |

All parameters live in `config.py` (`FingerprintConfig`, `MatchConfig`). `FingerprintConfig.version()` hashes them, so an index knows which config produced it.

## Ingestion pipeline

```
file ─▶ plan (sha256 + index-status lookup) ─▶ compute (worker processes) ─▶ commit (1 transaction/song)
        skip if every index is current           decode once per sample rate   insert song, replace stale rows,
                                                 run each Indexer.compute      write rows + song_indexes marker
```

- **Indexers** (`pipeline/indexers.py`) each declare a `name`, a `version` (a hash of their config) and the `sample_rate` they need. Phase 2 adds an `EmbeddingIndexer` next to `FingerprintIndexer`. Re-running `ingest` then backfills only the new index.
- **Idempotency** is tracked per `(song, index)` in `song_indexes`. The marker is committed in the same transaction as the index rows, so an interrupted run never leaves a half-indexed song. Songs are keyed by content hash, so a renamed copy isn't re-ingested.
- **Config changes** alter an indexer's version, and the next run replaces that index's rows for each song. A mixed-version index would silently break matching.

## Database schema

| Table | Purpose |
|---|---|
| `songs` | Library metadata; `content_hash` UNIQUE is the idempotency key |
| `song_indexes` | `(song_id, index_name) → version, item_count`: which indexes each song has, and with which config |
| `fingerprints` | `(hash INT, song_id INT, anchor_time INT)`, no primary key. A B-tree on `hash` for lookups; a BRIN index on `song_id` for re-indexing |

Why no primary key on `fingerprints`? The table is append-only and only ever read by `hash`. A surrogate key would add about 30% to the biggest table for nothing. Rows are bulk-loaded with **binary COPY**, with the payload built by numpy.

## Setup

Requires Python 3.11+, ffmpeg, and Docker.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
docker compose up -d                       # Postgres 16 on localhost:5433
.venv/bin/constellation db init

# Data: FMA metadata + a genre-balanced subset of fma_small, fetched via HTTP range
# requests (no 7 GB zip on disk). Re-runnable; --limit 8000 gets the full set.
.venv/bin/python scripts/download_fma.py --limit 2000

.venv/bin/constellation ingest data/fma_small --metadata data/fma_metadata/tracks.csv
.venv/bin/constellation db stats
.venv/bin/constellation identify path/to/clip.m4a
```

Set `CONSTELLATION_DATABASE_URL` to point at a different database.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest            # Postgres tests auto-skip if the compose DB isn't running
.venv/bin/pytest -m "not integration"
```
