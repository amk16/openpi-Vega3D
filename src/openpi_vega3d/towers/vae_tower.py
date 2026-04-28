"""VAE generative tower wrapping VEGA-3D's VAEOnlineEncoder."""

import logging
from types import SimpleNamespace

import torch
from torch import Tensor

from .base import BaseTower
from .rollout_tower_log import log_tower
from .vae_online_encoder import VAEOnlineEncoder

logger = logging.getLogger(__name__)


class VAETower(BaseTower):
    """Frozen SD2.1 VAE encoder that produces spatial latent tokens.

    Output shape: [B, output_spatial^2, 4] where output_spatial defaults to 14.
    So the default is [B, 196, 4].
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        input_size: int = 224,
        pool_kernel: int = 2,
        output_spatial: int = 14,
        dtype: str = "bf16",
    ):
        super().__init__()
        self._output_spatial = output_spatial
        self._feat_dim = 4

        config = SimpleNamespace(
            generative_vision_tower_checkpoint=checkpoint_dir,
            generative_vision_tower_input_size=input_size,
            generative_vision_tower_pool_kernel=pool_kernel,
            generative_vision_tower_output_spatial=output_spatial,
            generative_vision_tower_chunk_size=32,
            generative_vision_tower_dtype=dtype,
        )

        self.encoder = VAEOnlineEncoder(config)
        self.freeze()
        logger.info(
            "VAETower initialized (checkpoint=%s, output=[B, %d, %d])",
            checkpoint_dir, output_spatial * output_spatial, self._feat_dim,
        )
        log_tower(
            "VAETower ready: ckpt=%s tokens=%d feat_dim=%d dtype=%s",
            checkpoint_dir,
            output_spatial * output_spatial,
            self._feat_dim,
            dtype,
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def encode(self, images: Tensor) -> Tensor:
        latents = self.encoder(images)
        assert latents.ndim == 4, f"Expected [B,C,H,W] from encoder, got {tuple(latents.shape)}"
        b, c, h, w = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
