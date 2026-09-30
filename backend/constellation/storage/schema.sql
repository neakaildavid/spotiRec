-- Constellation schema. Idempotent: safe to run repeatedly (`constellation db init`).

CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, for Phase 2 similarity search

CREATE TABLE IF NOT EXISTS songs (
    id           SERIAL PRIMARY KEY,
    source       TEXT NOT NULL DEFAULT 'upload',
    source_id    TEXT,
    title        TEXT,
    artist       TEXT,
    album        TEXT,
    duration_s   REAL NOT NULL,
    file_path    TEXT NOT NULL,
    -- sha256 of the file bytes: the idempotency key (robust to renames/moves).
    content_hash TEXT NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per (song, index) once that index is fully written for the song.
-- Written in the same transaction as the index rows themselves.
CREATE TABLE IF NOT EXISTS song_indexes (
    song_id    INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    index_name TEXT NOT NULL,          -- 'fingerprint'; Phase 2 adds 'embedding'
    version    TEXT NOT NULL,          -- hash of the indexer config
    item_count INTEGER NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (song_id, index_name)
);

-- Append-only and read only by hash, so no surrogate primary key (it would add
-- ~30% to the size of the biggest table for no benefit).
CREATE TABLE IF NOT EXISTS fingerprints (
    hash        INTEGER NOT NULL,      -- packed (f1, f2, dt), < 2^31
    song_id     INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    anchor_time INTEGER NOT NULL       -- STFT frame index of the anchor peak
);

-- The lookup path: WHERE hash = ANY($1).
CREATE INDEX IF NOT EXISTS fingerprints_hash_idx ON fingerprints (hash);

-- Only for deleting/re-indexing one song. Rows are appended one song at a time,
-- so song_id tracks physical order and a BRIN index (a few KB, storing min/max
-- per block range) is enough, instead of a B-tree as large as the table's.
CREATE INDEX IF NOT EXISTS fingerprints_song_brin ON fingerprints USING brin (song_id);

-- ---------------------------------------------------------------- Phase 2

-- Top-level genre (FMA genre_top): shown in the UI and used by the similarity eval.
ALTER TABLE songs ADD COLUMN IF NOT EXISTS genre TEXT;

-- One "sounds like" vector per (song, model). The model name is part of the key
-- so embeddings from different models can coexist and be compared.
CREATE TABLE IF NOT EXISTS song_embeddings (
    song_id INTEGER NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    model   TEXT    NOT NULL,
    vec     vector  NOT NULL,          -- unit length, so cosine distance == 1 - dot
    PRIMARY KEY (song_id, model)
);

-- Approximate nearest-neighbour index (HNSW graph) for cosine distance.
-- The column is dimension-less so different models can share the table, but
-- an HNSW index needs a fixed dimension; hence one *partial expression* index
-- per model. Queries must use the same expression and WHERE clause to hit it.
CREATE INDEX IF NOT EXISTS song_embeddings_clap_htsat_hnsw ON song_embeddings
    USING hnsw ((vec::vector(512)) vector_cosine_ops)
    WHERE model = 'clap-htsat';
