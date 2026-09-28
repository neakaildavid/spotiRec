"""Evaluate identification accuracy, false positives, latency and DB size.

    python eval/run_eval.py --songs 200            # against the ingested Postgres library

Protocol
--------
* **Positives**: random library songs; for each clip length (5 s, 10 s) a random
  start is drawn, then every degradation is applied to that same clip, so
  conditions are compared on identical audio.
* **Negatives**: songs from ``data/fma_holdout`` that are *not* in the library
  (verified by content hash). They measure the false-positive rate: a query
  that should return "no match" but names a song.
* **Threshold calibration**: every query records its raw scores. The no-match
  threshold on confidence is chosen as the lowest value with zero false
  positives on the *calibration* half of the negatives, then accuracy and FPR are
  reported with it on the positives and on the *held-out test* half. Choosing
  and reporting on the same negatives would understate the FPR.
* **Accuracy** ("top-1") = the matcher returns a match *and* it's the right
  song. "Raw top-1" ignores the threshold (best candidate correct).
* **Latency** = fingerprinting + DB lookup + scoring for an already-decoded clip,
  measured sequentially on a warm DB after a short warm-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import add_white_noise, low_pass, mp3_roundtrip, phone_mic  # noqa: E402

from constellation.audio import load_audio  # noqa: E402
from constellation.config import FingerprintConfig, MatchConfig  # noqa: E402
from constellation.fingerprint import fingerprint  # noqa: E402
from constellation.matching import match_fingerprint  # noqa: E402
from constellation.metadata.fma import track_id_from_path  # noqa: E402
from constellation.pipeline.ingest import file_sha256  # noqa: E402
from constellation.storage.base import LookupResult  # noqa: E402
from constellation.storage.postgres import PostgresStorage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = FingerprintConfig().sample_rate

Degrade = Callable[[np.ndarray, np.random.Generator], np.ndarray]
CONDITIONS: dict[str, Degrade] = {
    "Clean": lambda x, rng: x,
    "White noise, 15 dB SNR": lambda x, rng: add_white_noise(x, 15, rng),
    "White noise, 5 dB SNR": lambda x, rng: add_white_noise(x, 5, rng),
    "White noise, 0 dB SNR": lambda x, rng: add_white_noise(x, 0, rng),
    "White noise, -5 dB SNR": lambda x, rng: add_white_noise(x, -5, rng),
    "Low-pass 3.4 kHz": lambda x, rng: low_pass(x, SR),
    "MP3 32 kbps": lambda x, rng: mp3_roundtrip(x, SR, "32k"),
    "Phone sim (band-pass + 10 dB noise + MP3)": lambda x, rng: phone_mic(x, SR, rng),
}


@dataclass
class QueryRecord:
    kind: str  # "positive" | "negative"
    source_track: str  # file name, for auditing individual queries
    split: str  # "test" | "calibration" (negatives only)
    condition: str
    clip_s: int
    true_song_id: int | None
    true_start_s: float
    best_song_id: int | None
    aligned: int
    runner_up: int
    confidence: float
    offset_s: float
    query_hashes: int
    db_hits: int
    t_fingerprint_ms: float
    t_lookup_ms: float
    t_score_ms: float

    @property
    def t_total_ms(self) -> float:
        return self.t_fingerprint_ms + self.t_lookup_ms + self.t_score_ms


class TimedStore:
    """Wraps a FingerprintStore to time the DB lookup separately from scoring."""

    def __init__(self, inner: PostgresStorage) -> None:
        self.inner = inner
        self.last_ms = 0.0

    def lookup(self, hashes: np.ndarray) -> LookupResult:
        t0 = time.perf_counter()
        out = self.inner.lookup(hashes)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out


def run_query(store: TimedStore, clip: np.ndarray, fp_cfg: FingerprintConfig) -> tuple:
    t0 = time.perf_counter()
    fp = fingerprint(clip, fp_cfg)
    t1 = time.perf_counter()
    store.last_ms = 0.0
    r = match_fingerprint(fp, store, fp_cfg, MatchConfig())
    t2 = time.perf_counter()
    match_ms = (t2 - t1) * 1000
    return r, (t1 - t0) * 1000, store.last_ms, max(0.0, match_ms - store.last_ms)


def load_exclusions(path: Path) -> dict[int, str]:
    """Held-out track ids known to contain library audio (see the file header)."""
    out: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                tid, _lib, reason = line.split(maxsplit=2)
                out[int(tid)] = reason
    return out


def build_queries(sources, clip_lengths, rng):
    """Yield (kind, split, song_id, path, clip_s, start_s) with starts drawn up front."""
    out = []
    for kind, split, song_id, path, dur in sources:
        for clip_s in clip_lengths:
            if dur < clip_s + 0.5:
                continue
            out.append((kind, split, song_id, path, clip_s, float(rng.uniform(0, dur - clip_s))))
    return out


def evaluate(args: argparse.Namespace) -> dict:
    fp_cfg = FingerprintConfig()
    rng = np.random.default_rng(args.seed)
    storage = PostgresStorage()
    store = TimedStore(storage)

    library = storage.list_songs(limit=10**7)
    picked = rng.choice(len(library), size=min(args.songs, len(library)), replace=False)
    sources = [("positive", "test", library[i].id, Path(library[i].file_path), library[i].duration_s) for i in picked]

    holdout = sorted(Path(args.holdout).rglob("*.mp3"))[: args.negatives]
    exclusions = load_exclusions(Path(args.exclusions))
    n_skipped_in_lib = 0
    excluded = []
    for j, path in enumerate(holdout):
        if storage.get_song_by_content_hash(file_sha256(path)) is not None:
            n_skipped_in_lib += 1  # guard: must never be in the library
            continue
        if track_id_from_path(path) in exclusions:
            excluded.append(f"{path.stem}: {exclusions[track_id_from_path(path)]}")
            continue
        split = "calibration" if j % 2 == 0 else "test"
        sources.append(("negative", split, None, path, 30.0))

    queries = build_queries(sources, args.clip_lengths, rng)

    # Warm-up: first queries pay for connection setup and cold DB pages.
    for q in queries[:10]:
        clip = load_audio(q[3], SR, start_s=q[5], duration_s=q[4])
        run_query(store, clip, fp_cfg)

    records: list[QueryRecord] = []
    audio_cache: dict[Path, np.ndarray] = {}
    for qi, (kind, split, song_id, path, clip_s, start_s) in enumerate(tqdm(queries, desc="queries", unit="clip")):
        if path not in audio_cache:
            audio_cache.clear()  # queries are grouped by song; keep one decoded song
            try:
                audio_cache[path] = load_audio(path, SR)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"skip {path.name}: {e}")
                continue
        audio = audio_cache.get(path)
        if audio is None:
            continue
        s = int(round(start_s * SR))
        base = audio[s : s + clip_s * SR]
        for ci, (cname, degrade) in enumerate(CONDITIONS.items()):
            qrng = np.random.default_rng([args.seed, qi, ci])
            clip = degrade(base, qrng)
            r, t_fp, t_lookup, t_score = run_query(store, clip, fp_cfg)
            best = r.candidates[0].song_id if r.candidates else None
            records.append(
                QueryRecord(kind, path.name, split, cname, clip_s, song_id, start_s, best, r.aligned_matches,
                            r.runner_up_matches, r.confidence, r.offset_s, r.query_hashes, r.db_hits,
                            t_fp, t_lookup, t_score)
            )

    stats = storage.stats()
    storage.close()
    return {"records": records, "stats": stats, "holdout_in_library": n_skipped_in_lib, "excluded": excluded,
            "n_positive_songs": len(picked), "n_negative_songs": sum(1 for s in sources if s[0] == "negative")}


# ---------------------------------------------------------------- analysis


def is_match(r: QueryRecord, threshold: float, min_aligned: int) -> bool:
    return r.best_song_id is not None and r.aligned >= min_aligned and r.confidence >= threshold


def calibrate(neg_cal: list[QueryRecord], min_aligned: int) -> float:
    """Lowest confidence threshold (0.01 grid) with zero calibration false positives."""
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        if not any(is_match(r, t, min_aligned) for r in neg_cal):
            return float(t)
    return 1.0


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def analyze(result: dict, args: argparse.Namespace) -> tuple[str, dict]:
    recs: list[QueryRecord] = result["records"]
    min_aligned = MatchConfig().min_aligned_matches
    pos = [r for r in recs if r.kind == "positive"]
    neg_cal = [r for r in recs if r.kind == "negative" and r.split == "calibration"]
    neg_test = [r for r in recs if r.kind == "negative" and r.split == "test"]
    thr = calibrate(neg_cal, min_aligned)
    lengths = sorted({r.clip_s for r in recs})

    def correct(r: QueryRecord) -> bool:
        return is_match(r, thr, min_aligned) and r.best_song_id == r.true_song_id

    lines: list[str] = []
    summary: dict = {"threshold": thr, "min_aligned": min_aligned, "accuracy": {}, "fpr": {}}

    # Accuracy table
    lines += ["### Top-1 accuracy (library songs)", ""]
    header = "| Condition | " + " | ".join(f"{L} s clip" for L in lengths) + " | " + " | ".join(f"{L} s raw top-1" for L in lengths) + " |"
    lines += [header, "|---|" + "---:|" * (2 * len(lengths))]
    for cname in CONDITIONS:
        acc_cells, raw_cells = [], []
        for L in lengths:
            rs = [r for r in pos if r.condition == cname and r.clip_s == L]
            acc_cells.append(pct(sum(correct(r) for r in rs), len(rs)))
            raw_cells.append(pct(sum(r.best_song_id == r.true_song_id for r in rs), len(rs)))
            summary["accuracy"][f"{cname} | {L}s"] = sum(correct(r) for r in rs) / max(1, len(rs))
        lines.append(f"| {cname} | " + " | ".join(acc_cells) + " | " + " | ".join(raw_cells) + " |")
    lines += ["", f"*Top-1* = returned a match (confidence ≥ {thr:.2f}, ≥ {min_aligned} aligned hashes) **and** it was the right song. "
              "*Raw top-1* ignores the no-match threshold.", ""]

    # False positives
    lines += ["### False-positive rate (songs not in the library)", "",
              "| Condition | " + " | ".join(f"{L} s clip" for L in lengths) + " |",
              "|---|" + "---:|" * len(lengths)]
    for cname in CONDITIONS:
        cells = []
        for L in lengths:
            rs = [r for r in neg_test if r.condition == cname and r.clip_s == L]
            cells.append(pct(sum(is_match(r, thr, min_aligned) for r in rs), len(rs)))
        lines.append(f"| {cname} | " + " | ".join(cells) + " |")
    fp_total = sum(is_match(r, thr, min_aligned) for r in neg_test)
    summary["fpr"]["overall"] = fp_total / max(1, len(neg_test))
    lines += ["", f"Overall: **{fp_total} / {len(neg_test)}** held-out queries ({pct(fp_total, len(neg_test))}) wrongly returned a song. "
              f"Threshold calibrated on a disjoint set of {len(neg_cal)} negative queries.", ""]

    # Threshold sweep
    lines += ["### Threshold trade-off (all conditions pooled)", "",
              "| Confidence threshold | Top-1 accuracy | False-positive rate |", "|---:|---:|---:|"]
    all_neg = neg_cal + neg_test
    for t in sorted({0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, thr}):
        acc = sum(is_match(r, t, min_aligned) and r.best_song_id == r.true_song_id for r in pos)
        fp = sum(is_match(r, t, min_aligned) for r in all_neg)
        mark = " ← chosen" if t == thr else ""
        lines.append(f"| {t:.2f}{mark} | {pct(acc, len(pos))} | {pct(fp, len(all_neg))} |")
    lines.append("")

    # Offset accuracy
    ok = [r for r in pos if correct(r)]
    errs = np.array([abs(r.offset_s - r.true_start_s) for r in ok])
    if errs.size:
        frame_ms = FingerprintConfig().frames_to_seconds(1) * 1000
        lines += ["### Offset accuracy (correct matches)", "",
                  f"Median |error| **{np.median(errs) * 1000:.0f} ms**, p95 {np.percentile(errs, 95) * 1000:.0f} ms; "
                  f"{pct(int((errs <= 0.1).sum()), errs.size)} within 100 ms (one STFT frame = {frame_ms:.0f} ms).", ""]
        summary["offset_median_ms"] = float(np.median(errs) * 1000)

    # Latency
    lines += ["### Query latency (decoded clip → result, warm DB)", "",
              "| Clip | Mean | p50 | p95 | Fingerprint | DB lookup | Scoring | Hashes / query | DB rows hit |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L in lengths:
        rs = [r for r in recs if r.clip_s == L]
        tot = np.array([r.t_total_ms for r in rs])
        lines.append(
            f"| {L} s | {tot.mean():.0f} ms | {np.percentile(tot, 50):.0f} ms | **{np.percentile(tot, 95):.0f} ms** | "
            f"{np.mean([r.t_fingerprint_ms for r in rs]):.0f} ms | {np.mean([r.t_lookup_ms for r in rs]):.0f} ms | "
            f"{np.mean([r.t_score_ms for r in rs]):.0f} ms | {np.mean([r.query_hashes for r in rs]):.0f} | "
            f"{np.mean([r.db_hits for r in rs]):,.0f} |"
        )
        summary[f"latency_{L}s_p95_ms"] = float(np.percentile(tot, 95))
    lines.append("")

    # DB size
    st = result["stats"]
    mb = lambda b: f"{b / 2**20:,.0f} MB"
    lines += ["### Index size", "",
              "| Songs | Fingerprint rows | Rows / song | Fingerprint table | Indexes | Whole DB |",
              "|---:|---:|---:|---:|---:|---:|",
              f"| {st['songs']:,} | {st['fingerprints']:,} | {st['fingerprints'] / max(1, st['songs']):,.0f} | "
              f"{mb(st['fingerprints_table_bytes'])} | {mb(st['fingerprints_index_bytes'])} | {mb(st['database_bytes'])} |", ""]

    # Setup
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = "?"
    cfg = FingerprintConfig()
    lines += [
        "### Setup", "",
        f"- {result['n_positive_songs']} library songs × {len(lengths)} clip lengths × {len(CONDITIONS)} conditions = {len(pos)} positive queries; "
        f"{result['n_negative_songs']} held-out songs → {len(all_neg)} negative queries.",
        f"- {len(result['excluded'])} held-out track(s) excluded after an audit found they contain library audio "
        f"(see `eval/shared_audio_exclusions.txt`): " + ("; ".join(result["excluded"]) or "none") + ".",
        f"- Library: {st['songs']:,} FMA tracks (30 s clips, 8 genres). Seed {args.seed}. Commit `{commit}`.",
        f"- Fingerprint config `{cfg.version()}`: {cfg.sample_rate} Hz, n_fft {cfg.n_fft}, hop {cfg.hop_length}, "
        f"{cfg.peaks_per_second} peaks/s, fan-out {cfg.fan_out}, target zone Δt {cfg.target_dt_min}–{cfg.target_dt_max} frames.",
        f"- Machine: {platform.machine()} / {platform.system()} {platform.release()}, Python {platform.python_version()}, "
        "Postgres 16 in Docker. Run " + datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".",
        "",
    ]
    return "\n".join(lines), summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", type=int, default=200, help="library songs to query")
    ap.add_argument("--negatives", type=int, default=200, help="held-out songs to query")
    ap.add_argument("--clip-lengths", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--holdout", default=str(ROOT / "data" / "fma_holdout"))
    ap.add_argument("--exclusions", default=str(ROOT / "eval" / "shared_audio_exclusions.txt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args()

    result = evaluate(args)
    md, summary = analyze(result, args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text("## Evaluation results\n\n" + md)
    (args.out / "results.json").write_text(json.dumps(
        {"summary": summary, "stats": result["stats"], "records": [asdict(r) for r in result["records"]]}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
