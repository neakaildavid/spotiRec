"""HNSW (approximate) vs exact nearest-neighbour search in pgvector.

    python eval/run_vector_index_eval.py

For every library song, the 10 most similar other songs are found three ways:
  * **numpy exact**: one matrix-vector product over all vectors (ground truth),
  * **pgvector HNSW** at several ``ef_search`` values (the production path),
  * **pgvector exact**: the same SQL with index scans disabled (sequential scan).

Recall@10 = |HNSW top-10 ∩ exact top-10| / 10. HNSW trades a little recall for
sub-linear search. At ~2k songs an exact scan is already fast, so this measures
the cost of being ready to scale, not a need at today's size.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from constellation.similarity import top_k
from constellation.storage.postgres import PostgresStorage, _vec_literal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "clap-htsat"
K = 10


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 100, 200])
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "vector_index.md")
    args = ap.parse_args()

    with PostgresStorage() as st:
        ids, X = st.all_embeddings(MODEL)
        n = len(ids)
        if n == 0:
            raise SystemExit(f"no '{MODEL}' embeddings; run: constellation ingest ... --indexes embedding")

        # Ground truth + numpy latency
        truth, np_ms = [], []
        for i in range(n):
            t0 = time.perf_counter()
            idx, _ = top_k(X[i], X, K, exclude=np.array([i]))
            np_ms.append((time.perf_counter() - t0) * 1000)
            truth.append(set(ids[idx].tolist()))

        # Warm the HNSW index pages
        for i in range(min(50, n)):
            st.nearest_embeddings(X[i], MODEL, K, exclude=(int(ids[i]),))

        rows = []
        for ef in args.ef:
            st.ef_search = ef
            rec, ms = [], []
            for i in range(n):
                t0 = time.perf_counter()
                hits = st.nearest_embeddings(X[i], MODEL, K, exclude=(int(ids[i]),))
                ms.append((time.perf_counter() - t0) * 1000)
                rec.append(len(truth[i] & {h[0] for h in hits}) / K)
            rows.append((ef, float(np.mean(rec)), float(np.mean(np.array(rec) == 1.0)), np.array(ms)))

        # Exact scan in SQL (index disabled) for the latency comparison
        exact_ms = []
        with st._conn() as conn:
            for i in range(n):
                q = _vec_literal(X[i])
                t0 = time.perf_counter()
                with conn.transaction():
                    conn.execute("SET LOCAL enable_indexscan = off")
                    conn.execute(
                        """SELECT song_id FROM song_embeddings WHERE model = %s
                           ORDER BY vec::vector(512) <=> %s::vector(512) LIMIT %s""",
                        (MODEL, q, K + 1),
                    ).fetchall()
                exact_ms.append((time.perf_counter() - t0) * 1000)
        exact_ms = np.array(exact_ms)
        stats = st.stats()

    L = ["## Vector index: HNSW vs exact search", "",
         f"{n:,} CLAP vectors (512-d). Every song queries its 10 nearest other songs; ground truth is exact search in numpy.", "",
         "| Method | `ef_search` | Recall@10 | Queries with perfect top-10 | p50 latency | p95 latency |",
         "|---|---:|---:|---:|---:|---:|"]
    for ef, r, perfect, ms in rows:
        mark = " (pgvector default)" if ef == 40 else ""
        L.append(f"| pgvector HNSW{mark} | {ef} | {100 * r:.1f}% | {100 * perfect:.1f}% | {pct(ms, 50):.1f} ms | {pct(ms, 95):.1f} ms |")
    L.append(f"| pgvector exact (seq scan) | – | 100% | 100% | {pct(exact_ms, 50):.1f} ms | {pct(exact_ms, 95):.1f} ms |")
    L.append(f"| numpy exact (in-process) | – | 100% | 100% | {pct(np.array(np_ms), 50):.2f} ms | {pct(np.array(np_ms), 95):.2f} ms |")
    L += ["",
          f"- **`ef_search` must be at least the query's LIMIT.** An HNSW scan returns at most `ef_search` rows. Each query asks for {K + 1} "
          f"(the top {K} plus the query song, which is filtered out), so `ef_search = 10` can return at most 9 real neighbours: "
          "recall is capped at 90% by construction, not by graph quality.",
          "- At this library size the whole index fits in memory and an exact scan is cheap. HNSW is here for the scaling path "
          "(its cost grows roughly logarithmically, a sequential scan's linearly).",
          "- Latencies are round trips from Python (including the query literal and the socket), so both SQL rows pay the same fixed overhead.", "",
          f"Storage: vectors {stats['embeddings_table_bytes'] / 2**20:.1f} MB, HNSW index {stats['embeddings_index_bytes'] / 2**20:.1f} MB "
          f"(vs {stats['fingerprints_total_bytes'] / 2**20:.0f} MB for the fingerprint index).", ""]
    md = "\n".join(L)
    args.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
