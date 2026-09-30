## Vector index: HNSW vs exact search

1,998 CLAP vectors (512-d). Every song queries its 10 nearest other songs; ground truth is exact search in numpy.

| Method | `ef_search` | Recall@10 | Queries with perfect top-10 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| pgvector HNSW | 10 | 89.9% | 0.0% | 1.2 ms | 1.8 ms |
| pgvector HNSW | 20 | 99.8% | 98.4% | 1.2 ms | 1.8 ms |
| pgvector HNSW (pgvector default) | 40 | 99.9% | 99.5% | 1.5 ms | 2.1 ms |
| pgvector HNSW | 100 | 100.0% | 99.9% | 1.7 ms | 2.8 ms |
| pgvector HNSW | 200 | 100.0% | 100.0% | 7.3 ms | 11.9 ms |
| pgvector exact (seq scan) | – | 100% | 100% | 3.9 ms | 5.0 ms |
| numpy exact (in-process) | – | 100% | 100% | 0.06 ms | 0.08 ms |

- **`ef_search` must be at least the query's LIMIT.** An HNSW scan returns at most `ef_search` rows. Each query asks for 11 (the top 10 plus the query song, which is filtered out), so `ef_search = 10` can return at most 9 real neighbours: recall is capped at 90% by construction, not by graph quality.
- At this library size the whole index fits in memory and an exact scan is cheap. HNSW is here for the scaling path (its cost grows roughly logarithmically, a sequential scan's linearly).
- Latencies are round trips from Python (including the query literal and the socket), so both SQL rows pay the same fixed overhead.

Storage: vectors 5.5 MB, HNSW index 5.3 MB (vs 248 MB for the fingerprint index).
