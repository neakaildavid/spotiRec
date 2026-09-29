"""Evaluate "sounds similar" embeddings (Phase 2).

    python eval/run_similarity_eval.py                 # CLAP vs hand-crafted vs random

There is no ground truth for perceived similarity, so we use standard proxies
from music information retrieval, all computed leave-one-out over the library:

* **Genre precision@10**: fraction of a song's 10 nearest neighbours that share
  its FMA top-level genre. 8 balanced genres, so random is about 12.5%.
* **Artist-filtered genre precision@10**: the same, but neighbours by the *same
  artist* are excluded first. Tracks from one artist/album sound alike and share
  a genre label, which inflates the plain metric (the "album effect").
* **kNN genre accuracy**: majority genre of the 10 neighbours == true genre.
* **Same-artist hit@10**: for songs whose artist has other tracks in the library,
  is at least one of them in the top 10?
* **Text -> audio** (CLAP only): zero-shot genre classification from prompts, and
  precision@10 of songs retrieved for "<genre> music".
* **Clip robustness**: a random 10 s clip (with noise) must retrieve its own song.
  This is the "sounds like..." fallback for songs identify can't match.

Embeddings are cached in data/embeddings_cache/, so re-runs only compute what's new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import add_white_noise, phone_mic  # noqa: E402

from constellation.audio import load_audio  # noqa: E402
from constellation.embedding import ClapEmbedder, HandcraftedEmbedder, l2_normalize, standardize  # noqa: E402
from constellation.metadata.fma import load_tracks  # noqa: E402
from constellation.similarity import top_k  # noqa: E402
from constellation.storage.postgres import PostgresStorage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embeddings_cache"
K = 10

# FMA's "International" is what most listeners (and CLAP's captions) call world music.
GENRE_PROMPT_NAME = {"International": "world", "Hip-Hop": "hip-hop"}
PROMPT_TEMPLATES = ["{} music", "a {} song", "this is a {} track", "an example of {} music"]


# ------------------------------------------------------------------ embeddings

def cached_embeddings(name: str, version: str, songs, compute) -> tuple[np.ndarray, dict]:
    """Return ``(matrix aligned with songs, timing)``; computes only uncached songs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}-{hashlib.sha1(version.encode()).hexdigest()[:8]}.npz"
    have: dict[int, np.ndarray] = {}
    cached_ms = float("nan")
    if path.exists():
        z = np.load(path)
        have = dict(zip(z["ids"].tolist(), z["vecs"]))
        cached_ms = float(z["ms"]) if "ms" in z else cached_ms
    todo = [s for s in songs if s.id not in have]
    times = []
    for s in tqdm(todo, desc=f"embed {name}", unit="song", disable=not todo):
        t0 = time.perf_counter()
        try:
            have[s.id] = compute(s)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"skip {s.id}: {e}")
            continue
        times.append(time.perf_counter() - t0)
        if len(times) % 200 == 0:
            save(path, have, float(np.mean(times) * 1000))
    ms = float(np.mean(times) * 1000) if times else cached_ms
    if todo:
        save(path, have, ms)
    mat = np.stack([have[s.id] for s in songs])
    return mat, {"ms_per_song": ms}


def save(path: Path, have: dict[int, np.ndarray], ms: float) -> None:
    """Persist vectors plus the measured per-song cost, so cached re-runs still report timing."""
    np.savez(path, ids=np.array(list(have)), vecs=np.stack(list(have.values())), ms=np.array(ms))


# ------------------------------------------------------------------ metrics

def neighbour_metrics(X: np.ndarray, genres: np.ndarray, artists: np.ndarray, rng: np.random.Generator | None = None) -> dict:
    """Leave-one-out retrieval metrics. With ``rng``, neighbours are random (the baseline)."""
    n = len(X)
    sims = None if rng is not None else X @ X.T
    artist_counts = Counter(artists.tolist())
    p10, p10_af, knn_ok, artist_hits, artist_queries = [], [], [], 0, 0
    per_genre: dict[str, list[float]] = {}
    for i in range(n):
        if rng is not None:
            order = rng.permutation(n)
            order = order[order != i]
        else:
            s = sims[i].copy()
            s[i] = -np.inf
            order = np.argsort(-s, kind="stable")
        top = order[:K]
        prec = float(np.mean(genres[top] == genres[i]))
        p10.append(prec)
        per_genre.setdefault(genres[i], []).append(prec)
        filtered = [j for j in order[: 5 * K] if artists[j] != artists[i]][:K]
        p10_af.append(float(np.mean(genres[filtered] == genres[i])) if filtered else 0.0)
        votes = Counter(genres[top].tolist()).most_common()
        knn_ok.append(votes[0][0] == genres[i])
        if artist_counts[artists[i]] > 1:
            artist_queries += 1
            artist_hits += bool(np.any(artists[top] == artists[i]))
    return {
        "genre_p10": float(np.mean(p10)),
        "genre_p10_artist_filtered": float(np.mean(p10_af)),
        "knn_genre_acc": float(np.mean(knn_ok)),
        "artist_hit10": artist_hits / max(1, artist_queries),
        "artist_queries": artist_queries,
        "per_genre_p10": {g: float(np.mean(v)) for g, v in sorted(per_genre.items())},
    }


def text_metrics(clap: ClapEmbedder, X: np.ndarray, genres: np.ndarray) -> dict:
    labels = sorted(set(genres.tolist()))
    # Prompt ensembling: average several phrasings per genre (standard CLIP/CLAP practice).
    prompt_vecs = []
    t0 = time.perf_counter()
    for g in labels:
        name = GENRE_PROMPT_NAME.get(g, g.lower())
        prompt_vecs.append(l2_normalize(clap.embed_text([t.format(name) for t in PROMPT_TEMPLATES]).mean(axis=0)))
    text_ms = (time.perf_counter() - t0) * 1000 / (len(labels) * len(PROMPT_TEMPLATES))
    P = np.stack(prompt_vecs)
    pred = np.array(labels)[np.argmax(X @ P.T, axis=1)]
    per_genre_acc = {g: float(np.mean(pred[genres == g] == g)) for g in labels}
    retrieval = {}
    for g, v in zip(labels, P):
        idx, _ = top_k(v, X, K)
        retrieval[g] = float(np.mean(genres[idx] == g))
    return {
        "zero_shot_acc": float(np.mean(pred == genres)),
        "zero_shot_per_genre": per_genre_acc,
        "text_retrieval_p10": float(np.mean(list(retrieval.values()))),
        "text_retrieval_per_genre": retrieval,
        "text_ms_per_prompt": text_ms,
    }


def clip_robustness(embed_clip, X: np.ndarray, songs, n: int, rng: np.random.Generator) -> dict:
    """Random 10 s clip (clean, 10 dB noise, phone chain) -> rank of its own song."""
    picks = rng.choice(len(songs), size=min(n, len(songs)), replace=False)
    out = {}
    conditions = {
        "clean": lambda x, r, sr: x,
        "noise 10 dB": lambda x, r, sr: add_white_noise(x, 10, r),
        "phone sim": lambda x, r, sr: phone_mic(x, sr, r),
    }
    ranks = {c: [] for c in conditions}
    for qi, i in enumerate(tqdm(picks, desc="clip queries", unit="clip")):
        audio_fn, sr = embed_clip["load"], embed_clip["sr"]
        audio = audio_fn(songs[i])
        start = int(rng.uniform(0, max(0, audio.size / sr - 10.5)) * sr)
        base = audio[start : start + 10 * sr]
        for c, deg in conditions.items():
            v = embed_clip["embed"](deg(base, np.random.default_rng([qi, len(ranks[c])]), sr))
            scores = X @ v
            ranks[c].append(int((scores > scores[i]).sum()) + 1)
    for c, r in ranks.items():
        r = np.array(r)
        out[c] = {"top1": float(np.mean(r == 1)), "top5": float(np.mean(r <= 5)), "median_rank": float(np.median(r))}
    return out


# ------------------------------------------------------------------ report

def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(res: dict) -> str:
    L = ["## Similarity evaluation (Phase 2)", ""]
    names = ["random", "handcrafted", "clap"]
    label = {"random": "Random", "handcrafted": "Hand-crafted (MFCC/chroma/contrast)", "clap": "**CLAP** (`clap-htsat-unfused`)"}
    L += ["### Retrieval quality (leave-one-out over the library, k = 10)", "",
          "| Embedding | Genre P@10 | Genre P@10, artist-filtered | kNN genre acc. | Same-artist hit@10 |",
          "|---|---:|---:|---:|---:|"]
    for nme in names:
        m = res["neighbours"][nme]
        L.append(f"| {label[nme]} | {pct(m['genre_p10'])} | {pct(m['genre_p10_artist_filtered'])} | {pct(m['knn_genre_acc'])} | {pct(m['artist_hit10'])} |")
    aq = res["neighbours"]["clap"]["artist_queries"]
    L += ["", f"Same-artist hit@10 is over the {aq} songs whose artist has other tracks in the library. "
          "Artist-filtered precision drops same-artist neighbours first, so the model can't score well just by finding the same album.", ""]

    L += ["### Genre P@10 by genre", "", "| Genre | Random | Hand-crafted | CLAP |", "|---|---:|---:|---:|"]
    for g in res["neighbours"]["clap"]["per_genre_p10"]:
        L.append(f"| {g} | " + " | ".join(pct(res["neighbours"][n]["per_genre_p10"][g]) for n in names) + " |")
    L.append("")

    t = res.get("text")
    if t:
        L += ["### Text → music (CLAP, zero-shot)", "",
              f"- Zero-shot genre classification from text prompts alone: **{pct(t['zero_shot_acc'])}** (chance 12.5%).",
              f"- Precision@10 of songs retrieved for \"<genre> music\": **{pct(t['text_retrieval_p10'])}**.",
              f"- Text embedding: {t['text_ms_per_prompt']:.0f} ms per prompt.", "",
              "| Genre | Zero-shot accuracy | Text retrieval P@10 |", "|---|---:|---:|"]
        for g in t["zero_shot_per_genre"]:
            L.append(f"| {g} | {pct(t['zero_shot_per_genre'][g])} | {pct(t['text_retrieval_per_genre'][g])} |")
        L.append("")
        if res.get("demo_queries"):
            L += ["Example free-text queries (top 3):", ""]
            for q, hits in res["demo_queries"].items():
                L.append(f"- *\"{q}\"* → " + "; ".join(f"{h['title']} ({h['genre']})" for h in hits))
            L.append("")

    L += ["### Clip → own song (the \"sounds like…\" fallback)", "",
          "A random 10 s clip is embedded and the library searched; how often is the clip's own song ranked first?", "",
          "| Condition | Hand-crafted top-1 | CLAP top-1 | CLAP top-5 | CLAP median rank |", "|---|---:|---:|---:|---:|"]
    for c in res["clips"]["clap"]:
        h, cl = res["clips"]["handcrafted"][c], res["clips"]["clap"][c]
        L.append(f"| {c} | {pct(h['top1'])} | {pct(cl['top1'])} | {pct(cl['top5'])} | {cl['median_rank']:.0f} |")
    L.append("")

    L += ["### Cost", "",
          f"- CLAP: **{res['timing']['clap'].get('ms_per_song', float('nan')):.0f} ms per 30 s song** (decode at 48 kHz + 3 windows on the Apple M2 GPU via MPS); 512 floats (2 KB) per song.",
          f"- Hand-crafted: {res['timing']['handcrafted'].get('ms_per_song', float('nan')):.0f} ms per song (CPU); 86 floats per song.",
          "", "### Setup", "",
          f"- {res['n_songs']:,} FMA tracks, 8 balanced top-level genres (genre labels from FMA metadata). Seed {res['seed']}.",
          "- Genre agreement is a *proxy* for perceived similarity: two tracks tagged \"Rock\" can sound nothing alike, and the Experimental and International labels are broad by nature.",
          ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with PostgresStorage() as st:
        songs = st.list_songs(limit=10**7)
    meta = load_tracks(ROOT / "data" / "fma_metadata" / "tracks.csv")
    songs = [s for s in songs if s.source_id and meta.get(int(s.source_id)) and meta[int(s.source_id)].genre]
    genres = np.array([meta[int(s.source_id)].genre for s in songs])
    artists = np.array([(s.artist or f"?{s.id}").strip().lower() for s in songs])

    clap = ClapEmbedder()
    hand = HandcraftedEmbedder()
    X_clap, t_clap = cached_embeddings("clap", clap.version, songs, lambda s: clap.embed_audio(load_audio(s.file_path, clap.sample_rate)))
    raw_hand, t_hand = cached_embeddings("handcrafted", hand.version, songs, lambda s: hand.raw_features(load_audio(s.file_path, hand.sample_rate)))
    mu, sd = raw_hand.mean(axis=0), raw_hand.std(axis=0)
    X_hand = standardize(raw_hand, mu, sd)

    res: dict = {"n_songs": len(songs), "seed": args.seed, "timing": {"clap": t_clap, "handcrafted": t_hand}}
    res["neighbours"] = {
        "random": neighbour_metrics(X_clap, genres, artists, rng=np.random.default_rng(args.seed)),
        "handcrafted": neighbour_metrics(X_hand, genres, artists),
        "clap": neighbour_metrics(X_clap, genres, artists),
    }
    res["text"] = text_metrics(clap, X_clap, genres)
    demo = ["mellow acoustic guitar with soft vocals", "aggressive distorted electric guitars and pounding drums",
            "upbeat electronic dance music with a heavy bassline", "slow ambient drone", "hip-hop beat with rap vocals"]
    res["demo_queries"] = {}
    for q in demo:
        idx, _ = top_k(clap.embed_text([q])[0], X_clap, 3)
        res["demo_queries"][q] = [{"title": songs[i].title, "artist": songs[i].artist, "genre": genres[i]} for i in idx]

    res["clips"] = {
        "clap": clip_robustness({"load": lambda s: load_audio(s.file_path, clap.sample_rate), "sr": clap.sample_rate,
                                 "embed": clap.embed_audio}, X_clap, songs, args.clips, np.random.default_rng(args.seed)),
        "handcrafted": clip_robustness({"load": lambda s: load_audio(s.file_path, hand.sample_rate), "sr": hand.sample_rate,
                                        "embed": lambda x: standardize(hand.raw_features(x)[None], mu, sd)[0]},
                                       X_hand, songs, args.clips, np.random.default_rng(args.seed)),
    }

    md = report(res)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "similarity.md").write_text(md)
    (args.out / "similarity.json").write_text(json.dumps(res, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()
