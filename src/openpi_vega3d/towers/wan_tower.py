"""WAN T2V generative tower wrapping VEGA-3D's WanT2VOnlineEncoder."""

import logging
from types import SimpleNamespace

import torch
from torch import Tensor

from .base import BaseTower
from .rollout_tower_log import log_tower
from .wan_t2v_encoder import WanT2VOnlineEncoder

logger = logging.getLogger(__name__)


class WanT2VTower(BaseTower):
    """Frozen WAN Text-to-Video encoder that produces spatial feature tokens.

    Output shape: [B, output_spatial**2, feat_dim] where feat_dim is the
    post-MLP residual-stream width at the hooked transformer block.
    Measured values (not the base `cfg.dim`): 1536 for the 1.3B variant.
    `cfg.dim` (1280 for 1.3B, 5120 for 14B) is the attention-working width,
    which differs from the residual-stream width the forward hook captures.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        prompt_emb_path: str | None = None,
        task: str = "t2v-1.3B",
        size: str = "832*480",
        timestep: int = 300,
        shift: float = 5.0,
        feat_block_idx: int = -1,
        output_spatial: int = 14,
        dtype: str = "bf16",
    ):
        super().__init__()

        config_kwargs = {
            "generative_vision_tower_checkpoint": checkpoint_dir,
            "generative_vision_tower_task": task,
            "generative_vision_tower_size": size,
            "generative_vision_tower_timestep": timestep,
            "generative_vision_tower_shift": shift,
            "generative_vision_tower_feat_block_idx": feat_block_idx,
            "generative_vision_tower_output_spatial": output_spatial,
            "generative_vision_tower_dtype": dtype,
        }
        if prompt_emb_path is not None:
            config_kwargs["generative_vision_tower_prompt_emb_path"] = prompt_emb_path

        config = SimpleNamespace(**config_kwargs)

        self.encoder = WanT2VOnlineEncoder(config)
        self._feat_dim = getattr(self.encoder.cfg, "dim", 1280)
        self._output_spatial = output_spatial
        self.freeze()
        logger.info(
            "WanT2VTower initialized (checkpoint=%s, task=%s, output=[B, 196, %d])",
            checkpoint_dir, task, self._feat_dim,
        )
        log_tower(
            "WanT2VTower ready: ckpt=%s task=%s feat_dim=%d prompt_emb=%s dtype=%s",
            checkpoint_dir,
            task,
            self._feat_dim,
            prompt_emb_path
            if prompt_emb_path is not None
            else "(encoder default path)",
            dtype,
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def encode(self, images: Tensor) -> Tensor:
        feats = self.encoder(images)
        assert feats.ndim == 4, f"Expected [B,C,H,W] from encoder, got {tuple(feats.shape)}"
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
