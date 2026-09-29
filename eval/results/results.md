## Evaluation results

### Top-1 accuracy (library songs)

| Condition | 5 s clip | 10 s clip | 5 s raw top-1 | 10 s raw top-1 |
|---|---:|---:|---:|---:|
| Clean | 100.0% | 100.0% | 100.0% | 100.0% |
| White noise, 15 dB SNR | 100.0% | 100.0% | 100.0% | 100.0% |
| White noise, 5 dB SNR | 99.0% | 100.0% | 100.0% | 100.0% |
| White noise, 0 dB SNR | 95.5% | 99.0% | 99.0% | 100.0% |
| White noise, -5 dB SNR | 74.5% | 92.0% | 96.0% | 97.5% |
| Low-pass 3.4 kHz | 100.0% | 100.0% | 100.0% | 100.0% |
| MP3 32 kbps | 100.0% | 100.0% | 100.0% | 100.0% |
| Phone sim (band-pass + 10 dB noise + MP3) | 100.0% | 100.0% | 100.0% | 100.0% |

*Top-1* = returned a match (confidence ≥ 0.25, ≥ 5 aligned hashes) **and** it was the right song. *Raw top-1* ignores the no-match threshold.

### False-positive rate (songs not in the library)

| Condition | 5 s clip | 10 s clip |
|---|---:|---:|
| Clean | 0.0% | 0.0% |
| White noise, 15 dB SNR | 0.0% | 0.0% |
| White noise, 5 dB SNR | 0.0% | 0.0% |
| White noise, 0 dB SNR | 0.0% | 0.0% |
| White noise, -5 dB SNR | 0.0% | 0.0% |
| Low-pass 3.4 kHz | 0.0% | 0.0% |
| MP3 32 kbps | 0.0% | 0.0% |
| Phone sim (band-pass + 10 dB noise + MP3) | 0.0% | 0.0% |

Overall: **0 / 1600** held-out queries (0.0%) wrongly returned a song. Threshold calibrated on a disjoint set of 1584 negative queries.

### Threshold trade-off (all conditions pooled)

| Confidence threshold | Top-1 accuracy | False-positive rate |
|---:|---:|---:|
| 0.00 | 99.5% | 58.3% |
| 0.05 | 99.4% | 11.4% |
| 0.10 | 99.2% | 2.6% |
| 0.15 | 98.8% | 1.1% |
| 0.20 | 98.5% | 0.2% |
| 0.25 ← chosen | 97.5% | 0.0% |
| 0.30 | 97.1% | 0.0% |
| 0.40 | 96.1% | 0.0% |
| 0.50 | 95.0% | 0.0% |

### Offset accuracy (correct matches)

Median |error| **9 ms**, p95 21 ms; 98.9% within 100 ms (one STFT frame = 23 ms).

### Query latency (decoded clip → result, warm DB)

| Clip | Mean | p50 | p95 | Fingerprint | DB lookup | Scoring | Hashes / query | DB rows hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 19 ms | 17 ms | **33 ms** | 5 ms | 13 ms | 1 ms | 344 | 6,544 |
| 10 s | 33 ms | 31 ms | **48 ms** | 10 ms | 21 ms | 1 ms | 709 | 12,817 |

### Index size

| Songs | Fingerprint rows | Rows / song | Fingerprint table | Indexes | Whole DB |
|---:|---:|---:|---:|---:|---:|
| 1,998 | 4,341,747 | 2,173 | 183 MB | 64 MB | 257 MB |

### Setup

- 200 library songs × 2 clip lengths × 8 conditions = 3200 positive queries; 199 held-out songs → 3184 negative queries.
- 1 held-out track(s) excluded after an audit found they contain library audio (see `eval/shared_audio_exclusions.txt`): 124992: Let Me Crazy "Intro" is reprised in "Outro" (same album); 72 aligned hashes at a constant 2.7 s offset across the whole clip.
- Library: 1,998 FMA tracks (30 s clips, 8 genres). Seed 0. Commit `9e3a8db`.
- Fingerprint config `fc1931cb812b`: 11025 Hz, n_fft 1024, hop 256, 15 peaks/s, fan-out 5, target zone Δt 1–63 frames.
- Machine: arm64 / Darwin 25.0.0, Python 3.13.9, Postgres 16 in Docker. Run 2026-09-28.
