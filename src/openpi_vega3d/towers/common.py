from __future__ import annotations

from contextlib import contextmanager
import math
from typing import List, Optional

import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def split_frames(frames: torch.Tensor, split_sizes: Optional[List[int]]) -> List[torch.Tensor]:
    if split_sizes is None:
        return [frames]
    if sum(split_sizes) != int(frames.shape[0]):
        raise ValueError(f"Invalid split_sizes {split_sizes} for frames with shape {tuple(frames.shape)}")
    if any(s <= 0 for s in split_sizes):
        raise ValueError(f"split_sizes must be positive, got {split_sizes}")
    return list(torch.split(frames, split_sizes, dim=0))


def to_unit_range(frames: torch.Tensor) -> torch.Tensor:
    x = frames.float()
    xmin = float(x.amin().item())
    xmax = float(x.amax().item())

    if xmin >= 0.0 and xmax <= 1.0:
        return x

    if xmin >= -1.05 and xmax <= 1.05:
        return (x + 1.0) * 0.5

    if xmin >= -4.5 and xmax <= 4.5:
        mean = x.new_tensor(CLIP_MEAN)[None, :, None, None]
        std = x.new_tensor(CLIP_STD)[None, :, None, None]
        x_clip = x * std + mean
        in_range = ((x_clip >= 0.0) & (x_clip <= 1.0)).float().mean()
        if float(in_range.item()) > 0.95:
            return x_clip.clamp(0.0, 1.0)

    x = x.clamp(-3.0, 3.0)
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def to_neg_one_to_one(frames: torch.Tensor) -> torch.Tensor:
    return to_unit_range(frames) * 2.0 - 1.0


def resize_center_crop(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(frames.shape)}")
    n, c, h, w = frames.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")

    scale = max(out_h / h, out_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    return x[:, :, top : top + out_h, left : left + out_w]


def resize_letterbox_pad(
    frames: torch.Tensor, out_h: int, out_w: int, pad_value: float = 0.0
) -> torch.Tensor:
    """Resize frames to fit inside (out_h, out_w) preserving aspect ratio, then
    pad the leftover border with `pad_value` (letterbox / pillarbox).

    Unlike `resize_center_crop`, no image content is discarded: the frame is
    scaled down to fit and the remaining border is filled. For a square frame
    going into a 16:9 target this produces vertical (pillarbox) bars.

    Frames are expected to already be in [-1, 1]; `pad_value=0.0` is mid-gray.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(frames.shape)}")
    n, c, h, w = frames.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")

    scale = min(out_h / h, out_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = max(0, out_h - new_h)
    pad_w = max(0, out_w - new_w)
    top = pad_h // 2
    left = pad_w // 2
    # F.pad pads the last dim first: (left, right, top, bottom).
    return F.pad(x, (left, pad_w - left, top, pad_h - top), mode="constant", value=pad_value)


def letterbox_content_box(
    h_in: int, w_in: int, out_h: int, out_w: int, px_per_token: int
) -> tuple[int, int, int, int]:
    """Content region of a letterboxed frame, in token coordinates.

    Mirrors `resize_letterbox_pad`'s geometry exactly (scale=min, centered
    padding) so the two cannot drift: for the same input/output sizes this
    returns the token-grid slice that covers every content token and excludes
    every pure-pad token. Boundaries are expanded outward (floor/ceil), so a
    token partially covered by content counts as content.

    For the LIBERO headline case (224x224 into 832x480, 16 px/token) the
    pillarbox is exactly 11 tokens per side, so the box is exact: rows 0:30,
    cols 11:41 — a clean 30x30 content square with zero partial tokens.

    Returns:
        (top, bottom, left, right) for slicing `feats[..., top:bottom, left:right]`
        on a [.., grid_h, grid_w] token grid.
    """
    scale = min(out_h / h_in, out_w / w_in)
    new_h = max(1, round(h_in * scale))
    new_w = max(1, round(w_in * scale))
    top_px = max(0, out_h - new_h) // 2
    left_px = max(0, out_w - new_w) // 2
    top = top_px // px_per_token
    left = left_px // px_per_token
    bottom = math.ceil((top_px + new_h) / px_per_token)
    right = math.ceil((left_px + new_w) / px_per_token)
    return top, bottom, left, right


def temporal_resample(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")
    if frames.shape[0] == target_frames:
        return frames
    if frames.shape[0] == 1:
        return frames.repeat(target_frames, 1, 1, 1)
    idx = torch.linspace(0, frames.shape[0] - 1, target_frames, device=frames.device)
    idx = idx.round().long().clamp(0, frames.shape[0] - 1)
    return frames[idx]


def resolve_inference_dtype(config, field_name: str = "generative_vision_tower_dtype") -> torch.dtype:
    """
    Resolve generative tower inference dtype from config.
    Defaults to bf16 on CUDA and fp32 on CPU.
    """
    raw_value = str(getattr(config, field_name, "bf16")).strip().lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if raw_value not in dtype_map:
        raise ValueError(
            f"Unsupported {field_name}={raw_value}. "
            "Expected one of: bf16|bfloat16|fp16|float16|fp32|float32."
        )
    dtype = dtype_map[raw_value]
    if not torch.cuda.is_available():
        return torch.float32
    return dtype


@contextmanager
def disable_hf_zero3_init():
    """
    Disable HuggingFace's global ZeRO-3-aware lazy init for ad-hoc runtime towers.
    Runtime generative towers are built after DeepSpeed engine init and are not managed
    by the engine, so loading them under HF ZeRO-3 context can leave partitioned params.
    """
    patched = []
    for module_name in (
        "transformers.integrations.deepspeed",
        "transformers.deepspeed",
        "transformers.modeling_utils",
    ):
        try:
            mod = __import__(module_name, fromlist=["is_deepspeed_zero3_enabled"])
            fn = getattr(mod, "is_deepspeed_zero3_enabled", None)
            if callable(fn):
                patched.append((mod, fn))
                setattr(mod, "is_deepspeed_zero3_enabled", lambda: False)
        except Exception:
            continue

    try:
        yield
    finally:
        for mod, fn in patched:
            setattr(mod, "is_deepspeed_zero3_enabled", fn)
