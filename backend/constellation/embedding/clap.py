"""CLAP audio/text embedder (LAION ``clap-htsat-unfused``).

CLAP (Contrastive Language-Audio Pretraining) trains an audio encoder and a
text encoder together so that a clip and a caption describing it land close in
one shared 512-d space. That gives us two features from one model:

* audio -> audio similarity ("songs that sound like this one"), and
* text -> audio search ("mellow acoustic guitar") with no extra training.

Checkpoint choice: ``laion/larger_clap_music`` looked like the obvious pick, but
its Hugging Face conversion is broken. Its contrastive temperature is stored as
~0.03 (exp -> 1.0, an untrained value), song embeddings are nearly parallel
(mean pairwise cosine 0.89), and zero-shot genre accuracy is at chance.
``clap-htsat-unfused`` is healthy on the same checks (temperature ~18.7, mean
cosine 0.50, zero-shot 35% on a balanced 80-song probe), so we use it.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from constellation.embedding.base import l2_normalize, split_windows

MODEL_ID = "laion/clap-htsat-unfused"
WINDOW_S = 10.0  # CLAP's training clip length


class ClapEmbedder:
    """Lazy-loading CLAP wrapper.

    The model (~0.8 GB) is loaded on first use, not at construction, so the
    object is cheap to create and to pickle, and importing this module doesn't
    require torch until an embedding is actually needed.
    """

    name = "clap-htsat"
    dim = 512
    sample_rate = 48_000  # what CLAP's feature extractor expects

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, max_windows: int = 12, batch_size: int = 12):
        self.model_id = model_id
        self.device = device
        self.max_windows = max_windows
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        # One inference at a time: the API calls this from a threadpool and a
        # single model instance on one GPU gains nothing from concurrent calls.
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return f"{self.model_id}|win{WINDOW_S:g}s|mean|v1"

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = state["_processor"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        # This checkpoint ships .bin weights; on every load transformers would
        # otherwise start a background thread asking the Hub to convert them to
        # safetensors, a network call even with local_files_only=True.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        import torch
        from transformers import ClapModel, ClapProcessor

        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer the local cache: otherwise every load makes (sometimes slow,
        # rate-limited) Hub requests to check for updates, even with the model
        # cached. Fall back to downloading on first use.
        try:
            self._processor = ClapProcessor.from_pretrained(self.model_id, local_files_only=True)
            model = ClapModel.from_pretrained(self.model_id, local_files_only=True)
        except OSError:
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            model = ClapModel.from_pretrained(self.model_id)
        self._model = model.eval().to(self.device)

    @staticmethod
    def _features(out: Any) -> Any:
        # transformers 5 returns an output object; 4.x returned the tensor itself.
        return out if hasattr(out, "shape") else out.pooler_output

    def embed_windows(self, samples: np.ndarray) -> np.ndarray:
        """Per-window embeddings, shape ``(n_windows, 512)``, each unit length."""
        import torch

        wins = split_windows(samples, self.sample_rate, WINDOW_S, self.max_windows)
        with self._lock:
            self._load()
            outs = []
            for i in range(0, len(wins), self.batch_size):
                # Windows are <= 10 s, so the feature extractor never applies its
                # random truncation: embeddings are deterministic.
                feats = self._processor(audio=wins[i : i + self.batch_size], sampling_rate=self.sample_rate, return_tensors="pt")
                feats = {k: v.to(self.device) for k, v in feats.items()}
                with torch.inference_mode():
                    outs.append(self._features(self._model.get_audio_features(**feats)).float().cpu().numpy())
        return l2_normalize(np.concatenate(outs))

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        """Song vector = normalized mean of its window vectors.

        Mean-pooling summarizes the whole track's sound (a song that is half
        piano ballad, half drums lands between the two), which is what
        "similar songs" should compare.
        """
        return l2_normalize(self.embed_windows(samples).mean(axis=0))

    def embed_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with self._lock:
            self._load()
            tok = self._processor(text=texts, return_tensors="pt", padding=True)
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.inference_mode():
                out = self._features(self._model.get_text_features(**tok)).float().cpu().numpy()
        return l2_normalize(out)
