"""Optional tower smoke checks for ``run_rollout`` and tooling."""

from __future__ import annotations

import torch
from torch import Tensor

from openpi_vega3d.towers.base import BaseTower
from openpi_vega3d.towers.rollout_tower_log import log_tower


def log_tower_registry_keys() -> None:
    from openpi_vega3d.towers import TOWER_REGISTRY

    log_tower("TOWER_REGISTRY keys: %s", list(TOWER_REGISTRY.keys()))


def run_base_tower_contract_smoke(*, device: torch.device | str = "cpu") -> dict:
    """Instantiate a minimal BaseTower and run check_output (no checkpoint)."""

    class _DummyTower(BaseTower):
        @property
        def feat_dim(self) -> int:
            return 8

        def encode(self, images: Tensor) -> Tensor:
            b = images.shape[0]
            return torch.randn(b, 196, self.feat_dim, device=images.device, dtype=images.dtype)

    log_tower("BaseTower contract smoke: DummyTower on device=%s", device)
    dev = torch.device(device) if isinstance(device, str) else device
    tower = _DummyTower()
    tower.freeze()
    images = torch.randn(1, 3, 224, 224, device=dev)
    result = tower.check_output(images)
    log_tower(
        "BaseTower contract smoke OK: out_shape=%s feat_dim=%s frozen=%s",
        result["output_shape"],
        result["feat_dim"],
        result["frozen"],
    )
    return result


def head_rgb_to_tower_input(head_rgb, *, device: torch.device) -> torch.Tensor:
    """Convert env observation ``head_rgb`` uint8 HWC to float [1,3,H,W] on device."""
    x = torch.as_tensor(head_rgb, device=device)
    if x.dtype != torch.float32:
        x = x.float()
    if x.ndim == 3:
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected HWC with C=3, got shape {tuple(x.shape)}")
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected HWC or NCHW image tensor, got {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 input channels, got {x.shape}")
    if x.max() > 1.5:
        x = x / 255.0
    return x


def run_tower_on_obs_head(
    tower: BaseTower,
    head_rgb,
    *,
    device: torch.device,
) -> dict:
    """Run ``check_output`` on the first camera image from a rollout observation."""
    log_tower(
        "gen_tower encode: tower=%s device=%s head_rgb shape=%s dtype=%s",
        tower.__class__.__name__,
        device,
        getattr(head_rgb, "shape", type(head_rgb)),
        getattr(head_rgb, "dtype", None),
    )
    images = head_rgb_to_tower_input(head_rgb, device=device)
    return tower.check_output(images)
