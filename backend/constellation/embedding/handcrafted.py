"""Hand-crafted audio features: the classic MIR baseline the learned model must beat.

Summary statistics (mean and std over time) of:
  * MFCCs (20)           : timbre, the "color" of the sound
  * chroma (12)          : harmony / pitch-class content
  * spectral contrast (7): peaky (tonal) vs flat (noisy) spectrum per band
  * centroid, bandwidth, rolloff, zero-crossing rate: brightness / noisiness

Raw features live on wildly different scales (MFCC0 is in the hundreds), so
cosine similarity is only meaningful after z-scoring each dimension with
library-wide statistics (:func:`standardize`). That makes this embedder need a
fitted scaler, one reason it's an eval baseline rather than a product index.
"""

from __future__ import annotations

import numpy as np

from constellation.embedding.base import l2_normalize


class HandcraftedEmbedder:
    name = "handcrafted-v1"
    dim = 86  # 43 features x (mean, std)
    sample_rate = 22_050

    @property
    def version(self) -> str:
        return "mfcc20+chroma12+contrast7+spectral4|mean,std|v1"

    def raw_features(self, samples: np.ndarray) -> np.ndarray:
        import librosa

        y = np.asarray(samples, np.float32)
        sr = self.sample_rate
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(S=S**2, sr=sr)
        feats = [
            librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20),
            librosa.feature.chroma_stft(S=S, sr=sr),
            librosa.feature.spectral_contrast(S=S, sr=sr),
            librosa.feature.spectral_centroid(S=S, sr=sr),
            librosa.feature.spectral_bandwidth(S=S, sr=sr),
            librosa.feature.spectral_rolloff(S=S, sr=sr),
            librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512),
        ]
        m = np.concatenate(feats, axis=0)  # (43, frames)
        return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        """Unscaled features, L2-normalized. Prefer :func:`standardize` over a library."""
        return l2_normalize(self.raw_features(samples))


def standardize(raw: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray:
    """Z-score each feature dimension (library statistics), then L2-normalize rows."""
    mean = raw.mean(axis=0) if mean is None else mean
    std = raw.std(axis=0) if std is None else std
    return l2_normalize((raw - mean) / np.maximum(std, 1e-8))
