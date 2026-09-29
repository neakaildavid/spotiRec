"""POST /identify: fingerprint an uploaded clip and match it against the library."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from constellation.api.deps import get_settings, get_storage, read_upload, upload_suffix
from constellation.api.schemas import IdentifyResponse, QueryInfo, SongOut, Timing
from constellation.api.settings import Settings
from constellation.audio import load_audio_bytes
from constellation.fingerprint import fingerprint
from constellation.matching import match_fingerprint
from constellation.storage.base import Storage

router = APIRouter(tags=["identify"])

_MAX_PEAKS = 1000


@router.post("/identify", response_model=IdentifyResponse)
def identify(
    request: Request,
    file: UploadFile = File(..., description="Audio clip in any format ffmpeg can decode"),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> IdentifyResponse:
    """Identify a clip. Returns ``match: false`` (HTTP 200) when nothing is
    confident enough; "no match" is a normal answer, not an error.

    Defined as a sync ``def`` on purpose: decoding and fingerprinting are CPU work
    and the DB driver is synchronous, so FastAPI runs this in its threadpool
    rather than blocking the event loop.
    """
    fp_cfg = request.app.state.fp_config
    data = read_upload(file, settings.max_identify_bytes)

    t0 = time.perf_counter()
    samples = load_audio_bytes(data, fp_cfg.sample_rate, suffix=upload_suffix(file))
    samples = samples[: int(settings.max_query_seconds * fp_cfg.sample_rate)]
    duration_s = samples.size / fp_cfg.sample_rate
    if duration_s < settings.min_query_seconds:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"clip too short ({duration_s:.1f} s)")

    t1 = time.perf_counter()
    fp = fingerprint(samples, fp_cfg)
    t2 = time.perf_counter()
    result = match_fingerprint(fp, storage, fp_cfg, request.app.state.match_config)
    song = storage.get_song(result.song_id) if result.is_match and result.song_id is not None else None
    t3 = time.perf_counter()

    hz_per_bin = fp_cfg.sample_rate / fp_cfg.n_fft
    step = max(1, fp.peak_times.size // _MAX_PEAKS)
    peaks = [
        (round(fp_cfg.frames_to_seconds(int(t)), 3), round(float(f) * hz_per_bin, 1))
        for t, f in zip(fp.peak_times[::step], fp.peak_freqs[::step])
    ]
    ms = lambda a, b: round((b - a) * 1000, 1)
    return IdentifyResponse(
        match=song is not None,
        song=SongOut.from_song(song) if song else None,
        offset_s=round(max(0.0, result.offset_s), 3) if song else None,
        confidence=round(result.confidence, 3),
        aligned_matches=result.aligned_matches,
        runner_up_matches=result.runner_up_matches,
        query=QueryInfo(duration_s=round(duration_s, 2), hashes=len(fp), peaks=peaks),
        timing=Timing(decode_ms=ms(t0, t1), fingerprint_ms=ms(t1, t2), match_ms=ms(t2, t3), total_ms=ms(t0, t3)),
    )
