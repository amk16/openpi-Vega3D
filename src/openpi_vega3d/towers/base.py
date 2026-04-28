"""Abstract base class for generative vision towers."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from openpi_vega3d.towers.rollout_tower_log import log_tower


class BaseTower(nn.Module, ABC):
    """Frozen generative encoder that extracts spatial features from images.

    All towers follow the same contract:
        Input:  [B, 3, H, W] float tensor (any range -- tower handles normalization)
        Output: [B, num_tokens, feat_dim] float tensor

    Towers are always frozen (requires_grad=False).  The projector that maps
    tower features to the policy's embedding dimension lives in SpatialEnhancer.
    """

    @abstractmethod
    def encode(self, images: Tensor) -> Tensor:
        """Encode images to spatial feature tokens.

        Args:
            images: [B, 3, H, W] input images.

        Returns:
            [B, num_tokens, feat_dim] feature tokens.
        """

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        """Dimensionality of each output token."""

    def forward(self, images: Tensor) -> Tensor:
        return self.encode(images)

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def check_output(self, images: Tensor) -> dict:
        """Diagnostic: run encode and return shape/stats."""
        log_tower(
            "check_output start: %s in_shape=%s device=%s",
            type(self).__name__,
            tuple(images.shape),
            images.device,
        )
        out = self.encode(images)
        frozen = all(not p.requires_grad for p in self.parameters())
        result = {
            "input_shape": tuple(images.shape),
            "output_shape": tuple(out.shape),
            "feat_dim": self.feat_dim,
            "mean": float(out.mean()),
            "std": float(out.std()),
            "frozen": frozen,
        }
        log_tower(
            "check_output done: %s out_shape=%s mean=%.4f std=%.4f frozen=%s",
            type(self).__name__,
            result["output_shape"],
            result["mean"],
            result["std"],
            frozen,
        )
        return result
