"""Download FMA metadata and a genre-balanced subset of fma_small.

Instead of downloading the 7.2 GB fma_small.zip, this reads the zip's central
directory over HTTP range requests and fetches only the chosen members, so the
disk only ever holds the extracted mp3s we actually use.

    python scripts/download_fma.py --limit 2000        # balanced subset
    python scripts/download_fma.py --limit 8000        # all of fma_small

Re-running is incremental: files already on disk are skipped.
"""

from __future__ import annotations

import argparse
import bz2
import io
import random
import struct
import sys
import zlib
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from constellation.metadata.fma import audio_relpath, load_tracks  # noqa: E402

BASE_URL = "https://os.unil.cloud.switch.ch/fma"
DATA = Path(__file__).resolve().parents[1] / "data"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(name: str) -> zipfile.ZipFile:
    """Open a remote zip; only its central directory (the file index) is fetched."""
    raw = HttpRangeFile(f"{BASE_URL}/{name}")
    # Small buffer: only used while reading the central directory. Member data is
    # fetched with one exact-range request each (see fetch_member).
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=256 << 10))


def fetch_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Fetch one member with a single Range request covering header + data.

    Going through ``zf.open`` would issue several small reads (local header,
    then data in chunks), and any buffer is discarded on every seek between
    members. One exact request per file is ~4x less traffic and far fewer
    round trips.
    """
    raw: HttpRangeFile = zf.fp.raw  # type: ignore[union-attr]
    # Local header is 30 bytes + name + extra field; allow slack for the extra field.
    span = 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size
    raw.seek(info.header_offset)
    buf = bytearray(min(span, raw.size - info.header_offset))
    buf = buf[: raw.readinto(buf)]
    sig, *_rest = struct.unpack("<4s22xHH", bytes(buf[:30]))
    if sig != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", bytes(buf[26:30]))
    start = 30 + name_len + extra_len
    data = bytes(buf[start : start + info.compress_size])
    if len(data) != info.compress_size:
        raise ValueError(f"short read for {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    elif info.compress_type == zipfile.ZIP_BZIP2:  # what fma_small.zip uses
        out = bz2.decompress(data)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(out) != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return out


def extract(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(fetch_member(zf, zf.getinfo(member)))
    tmp.rename(dest)  # atomic: a crash never leaves a truncated .mp3 behind


def fetch_metadata() -> Path:
    meta_dir = DATA / "fma_metadata"
    for name in ("tracks.csv", "genres.csv"):
        dest = meta_dir / name
        if not dest.exists():
            print(f"fetching fma_metadata/{name} ...", flush=True)
            extract(open_remote_zip("fma_metadata.zip"), f"fma_metadata/{name}", dest)
    return meta_dir / "tracks.csv"


def choose_tracks(tracks_csv: Path, limit: int, seed: int) -> list[int]:
    """Pick ``limit`` fma_small tracks, round-robin across genres (deterministic)."""
    by_genre: dict[str, list[int]] = defaultdict(list)
    for t in load_tracks(tracks_csv).values():
        if t.subset == "small":
            by_genre[t.genre or "?"].append(t.track_id)
    rng = random.Random(seed)
    for ids in by_genre.values():
        ids.sort()
        rng.shuffle(ids)
    chosen: list[int] = []
    genres = sorted(by_genre)
    i = 0
    while len(chosen) < limit and any(i < len(by_genre[g]) for g in genres):
        chosen += [by_genre[g][i] for g in genres if i < len(by_genre[g])]
        i += 1
    return chosen[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tracks_csv = fetch_metadata()
    wanted = choose_tracks(tracks_csv, args.limit, args.seed)
    out_dir = DATA / "fma_small"
    todo = [tid for tid in wanted if not (out_dir / audio_relpath(tid)).exists()]
    print(f"{len(wanted)} tracks selected, {len(wanted) - len(todo)} already present, {len(todo)} to fetch")
    if not todo:
        return

    # One ZipFile (and HTTP stream) per worker: ZipFile objects aren't thread-safe.
    chunks = [todo[i :: args.workers] for i in range(args.workers)]

    def work(ids: list[int]) -> int:
        zf = open_remote_zip("fma_small.zip")
        for tid in ids:
            rel = audio_relpath(tid)
            extract(zf, f"fma_small/{rel}", out_dir / rel)
            progress()
        return len(ids)

    done = 0

    def progress() -> None:
        nonlocal done
        done += 1
        if done % 50 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(work, c) for c in chunks if c]):
            fut.result()
    print("done")


if __name__ == "__main__":
    main()
