"""Audio embedders for similarity search (Phase 2).

``ClapEmbedder`` needs the optional ``embeddings`` extra (torch, transformers);
it imports them lazily so the rest of the package works without them.
"""

from constellation.embedding.base import Embedder, TextEmbedder, l2_normalize, split_windows
from constellation.embedding.clap import ClapEmbedder
from constellation.embedding.handcrafted import HandcraftedEmbedder, standardize

__all__ = [
    "ClapEmbedder",
    "Embedder",
    "HandcraftedEmbedder",
    "TextEmbedder",
    "l2_normalize",
    "split_windows",
    "standardize",
]
