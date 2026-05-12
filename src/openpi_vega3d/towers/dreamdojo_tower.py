"""DreamDojo generative tower wrapping Cosmos-Predict2.5-2B.

Phase 6 backbone: extracts intermediate DiT features from a frozen
Cosmos-Predict2.5-2B transformer via a single denoising step, analogous
to how WanT2VTower extracts features from WAN T2V.

Sub-phase 6.2: real model loader with feat_dim introspection.
encode() still returns dummy zeros — real forward comes in 6.3.
"""

import logging
import os

import torch
from torch import Tensor

from .base import BaseTower

logger = logging.getLogger(__name__)

# Cosmos-Predict2.5-2B architecture config (confirmed via DreamDojo DCP metadata).
# 16 heads * 128 dim = 2048 hidden_size, 28 transformer blocks, 1.96B params.
# in_channels=17: DreamDojo adds 1 action channel on top of base Cosmos's 16 VAE
# latent channels. With concat_padding_mask=True, total patchify input = 18 channels,
# matching DreamDojo's x_embedder.proj.1.weight shape of (2048, 72) = (hidden, 18*4).
_COSMOS_2B_CONFIG = {
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "in_channels": 17,
    "out_channels": 16,
    "patch_size": (1, 2, 2),
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "extra_pos_embed_type": "learnable",
}

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_DREAMDOJO_ACTION_KEY_PREFIXES = (
    "action_embedder_",
)


class DreamDojoTower(BaseTower):
    """Frozen Cosmos-Predict2.5-2B encoder that produces spatial feature tokens.

    Output shape: [B, output_spatial**2, feat_dim].
    feat_dim = num_attention_heads * attention_head_dim = 2048 for 2B.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        variant: str = "teacher",
        input_resolution: int = 256,
        timestep: int = 300,
        feat_block_idx: int = 20,
        output_spatial: int = 16,
        dtype: str = "bf16",
    ):
        super().__init__()

        if variant == "student":
            raise NotImplementedError(
                "variant='student' is not supported in Phase 6. The 4-step distilled "
                "student does not support intermediate-noise feature extraction. "
                "Use variant='teacher'."
            )

        self._checkpoint_dir = checkpoint_dir
        self._variant = variant
        self._input_resolution = input_resolution
        self._timestep = timestep
        self._feat_block_idx = feat_block_idx
        self._output_spatial = output_spatial
        self._dtype_str = dtype

        ckpt_path = _find_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            self.transformer = _load_transformer(ckpt_path, dtype)
            cfg = self.transformer.config
            self._feat_dim = cfg.num_attention_heads * cfg.attention_head_dim
            self._num_blocks = len(self.transformer.transformer_blocks)
        else:
            self.transformer = None
            self._feat_dim = (
                _COSMOS_2B_CONFIG["num_attention_heads"]
                * _COSMOS_2B_CONFIG["attention_head_dim"]
            )
            self._num_blocks = _COSMOS_2B_CONFIG["num_layers"]
            logger.warning(
                "DreamDojoTower: no checkpoint at %s — offline mode (dummy features). "
                "Download DreamDojo 2B pretrain from nvidia/DreamDojo on HuggingFace, "
                "convert DCP to .pt, place in checkpoint_dir.",
                checkpoint_dir,
            )

        if self._feat_block_idx >= self._num_blocks:
            raise ValueError(
                f"feat_block_idx={self._feat_block_idx} >= num_blocks={self._num_blocks}"
            )

        self.freeze()
        logger.info(
            "DreamDojoTower ready (checkpoint=%s, variant=%s, res=%d, "
            "block=%d/%d, spatial=%d, feat_dim=%d, online=%s)",
            checkpoint_dir,
            variant,
            input_resolution,
            feat_block_idx,
            self._num_blocks,
            output_spatial,
            self._feat_dim,
            ckpt_path is not None,
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def encode(self, images: Tensor) -> Tensor:
        b = images.shape[0]
        num_tokens = self._output_spatial ** 2
        return torch.zeros(
            b, num_tokens, self._feat_dim,
            dtype=images.dtype, device=images.device,
        )


def _find_checkpoint(checkpoint_dir: str) -> str | None:
    """Locate a DreamDojo .pt checkpoint file in checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        return None

    for name in ("model_ema_bf16.pt", "model.pt", "dreamdojo_2b.pt"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            return path

    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)

    return None


def _load_transformer(ckpt_path: str, dtype: str):
    """Instantiate CosmosTransformer3DModel and load DreamDojo weights.

    Uses diffusers' built-in Cosmos key conversion to map NVIDIA-native
    key names (net.blocks.*, net.x_embedder.*, etc.) to diffusers format
    (transformer_blocks.*, patch_embed.*, etc.). DreamDojo's extra
    action-conditioning keys are skipped via strict=False.
    """
    from diffusers.models import CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )

    pt_dtype = _DTYPE_MAP[dtype]

    transformer = CosmosTransformer3DModel(**_COSMOS_2B_CONFIG)

    logger.info("Loading DreamDojo checkpoint from %s ...", ckpt_path)
    try:
        raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        logger.warning("Loaded checkpoint with weights_only=False (non-standard tensors).")

    converted_sd = convert_cosmos_transformer_checkpoint_to_diffusers(raw_sd)
    result = transformer.load_state_dict(converted_sd, strict=False)

    action_keys = [
        k for k in result.unexpected_keys
        if any(k.startswith(p) for p in _DREAMDOJO_ACTION_KEY_PREFIXES)
    ]
    other_unexpected = [
        k for k in result.unexpected_keys if k not in set(action_keys)
    ]

    loaded_count = len(converted_sd) - len(result.unexpected_keys)
    logger.info(
        "DreamDojo loaded: %d matched, %d missing, %d action keys skipped, %d other unexpected",
        loaded_count,
        len(result.missing_keys),
        len(action_keys),
        len(other_unexpected),
    )
    if result.missing_keys:
        logger.info("Missing keys (first 10): %s", result.missing_keys[:10])
    if other_unexpected:
        logger.warning("Unexpected non-action keys (first 10): %s", other_unexpected[:10])

    return transformer.to(dtype=pt_dtype)
