## Similarity evaluation (Phase 2)

### Retrieval quality (leave-one-out over the library, k = 10)

| Embedding | Genre P@10 | Genre P@10, artist-filtered | kNN genre acc. | Same-artist hit@10 |
|---|---:|---:|---:|---:|
| Random | 12.8% | 12.6% | 13.7% | 3.0% |
| Hand-crafted (MFCC/chroma/contrast) | 35.5% | 32.4% | 47.4% | 36.9% |
| **CLAP** (`clap-htsat-unfused`) | 52.0% | 48.0% | 63.6% | 54.6% |

Same-artist hit@10 is over the 1361 songs whose artist has other tracks in the library. Artist-filtered precision drops same-artist neighbours first, so the model can't score well just by finding the same album.

### Genre P@10 by genre

| Genre | Random | Hand-crafted | CLAP |
|---|---:|---:|---:|
| Electronic | 12.1% | 31.6% | 55.1% |
| Experimental | 11.7% | 25.8% | 31.6% |
| Folk | 12.7% | 43.2% | 58.8% |
| Hip-Hop | 12.1% | 49.2% | 70.4% |
| Instrumental | 14.0% | 41.6% | 50.4% |
| International | 14.6% | 32.8% | 58.8% |
| Pop | 12.3% | 13.6% | 32.0% |
| Rock | 12.8% | 46.0% | 58.7% |

### Text → music (CLAP, zero-shot)

- Zero-shot genre classification from text prompts alone: **35.9%** (chance 12.5%).
- Precision@10 of songs retrieved for "<genre> music": **35.0%**.
- Text embedding: 296 ms per prompt.

| Genre | Zero-shot accuracy | Text retrieval P@10 |
|---|---:|---:|
| Electronic | 24.4% | 50.0% |
| Experimental | 3.2% | 0.0% |
| Folk | 62.8% | 20.0% |
| Hip-Hop | 78.4% | 90.0% |
| Instrumental | 16.0% | 80.0% |
| International | 2.0% | 0.0% |
| Pop | 17.6% | 10.0% |
| Rock | 82.7% | 30.0% |

Example free-text queries (top 3):

- *"mellow acoustic guitar with soft vocals"* → I Had A Lover I Thought Was My Own (Folk); Mosic (Instrumental); Povo Que Caís Descalço (International)
- *"aggressive distorted electric guitars and pounding drums"* → Come To My Coast (Instrumental) (Folk); Bugger Me (Pop); Alarm (Instrumental)
- *"upbeat electronic dance music with a heavy bassline"* → Ghost By Nature - Pushing Situation (Electronic); Nihilist (Pop); Blue Pill (Electronic)
- *"slow ambient drone"* → IV (Instrumental); Emptiness (Instrumental); Under the Color Cave (Experimental)
- *"hip-hop beat with rap vocals"* → 05 Too Complex Instrumental     {6th Sense & The Kid Daytona}.mp3 (Hip-Hop); Te quiero (Folk); Albayan (International)

### Clip → own song (the "sounds like…" fallback)

A random 10 s clip is embedded and the library searched; how often is the clip's own song ranked first?

| Condition | Hand-crafted top-1 | CLAP top-1 | CLAP top-5 | CLAP median rank |
|---|---:|---:|---:|---:|
| clean | 95.0% | 97.5% | 99.0% | 1 |
| noise 10 dB | 3.5% | 37.5% | 59.5% | 4 |
| phone sim | 0.5% | 4.0% | 13.5% | 60 |

### Cost

- CLAP: **244 ms per 30 s song** (decode at 48 kHz + 3 windows on the Apple M2 GPU via MPS); 512 floats (2 KB) per song.
- Hand-crafted: 139 ms per song (CPU); 86 floats per song.

### Setup

- 1,998 FMA tracks, 8 balanced top-level genres (genre labels from FMA metadata). Seed 0.
- Genre agreement is a *proxy* for perceived similarity: two tracks tagged "Rock" can sound nothing alike, and the Experimental and International labels are broad by nature.
