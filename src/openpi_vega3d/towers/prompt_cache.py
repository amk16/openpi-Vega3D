"""Precomputed T5 prompt embedding cache for Cosmos towers.

Stores a mapping from prompt string → [seq_len, embed_dim] tensor on disk.
At runtime, loads the full cache into memory for O(1) lookup by prompt.
Raises KeyError for prompts not in the cache — re-run the export script to add them.
"""

from __future__ import annotations

import pathlib

import torch
from torch import Tensor


class PromptEmbeddingCache:
    """Lookup table of precomputed T5 prompt embeddings.

    Args:
        cache_path: Path to a .pt file saved by export_cosmos_prompt_embeddings.py.
            The file contains a dict with:
                "embeddings": {prompt_str: Tensor[seq_len, embed_dim]}
                "embed_dim": int
                "seq_len": int
    """

    def __init__(self, cache_path: str | pathlib.Path) -> None:
        cache_path = pathlib.Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Prompt embedding cache not found at {cache_path}. "
                "Run scripts/export_cosmos_prompt_embeddings.py first."
            )
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self._embeddings: dict[str, Tensor] = data["embeddings"]
        self._embed_dim: int = data["embed_dim"]
        self._seq_len: int = data["seq_len"]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def prompts(self) -> list[str]:
        return list(self._embeddings.keys())

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._embeddings

    def __getitem__(self, prompt: str) -> Tensor:
        """Return the cached [seq_len, embed_dim] embedding for a prompt.

        Raises:
            KeyError: If the prompt is not in the cache.
        """
        if prompt not in self._embeddings:
            raise KeyError(
                f"Prompt not in cache: {prompt!r}. "
                f"Cache contains {len(self._embeddings)} prompts. "
                "Re-run scripts/export_cosmos_prompt_embeddings.py with the updated dataset."
            )
        return self._embeddings[prompt]

    def get_batch(self, prompts: list[str], *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Return a batched [B, seq_len, embed_dim] tensor for a list of prompts."""
        embeds = torch.stack([self[p] for p in prompts], dim=0)
        if dtype is not None:
            embeds = embeds.to(dtype=dtype)
        if device is not None:
            embeds = embeds.to(device=device)
        return embeds
