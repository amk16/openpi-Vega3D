"""DreamDojo generative tower wrapping Cosmos-Predict2.5-2B.

Phase 6 backbone: extracts intermediate DiT features from a frozen
Cosmos-Predict2.5-2B transformer via a single denoising step, analogous
to how WanT2VTower extracts features from WAN T2V.

Sub-phase 6.3: real encode() with null-text forward pass through the
Cosmos DiT, hooked at an intermediate transformer block.
"""

import logging
import os

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812 — project convention

from .base import BaseTower
from .common import to_neg_one_to_one

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

_DREAMDOJO_ACTION_KEY_PREFIXES = ("action_embedder_",)


class DreamDojoTower(BaseTower):
    """Frozen Cosmos-Predict2.5-2B encoder that produces spatial feature tokens.

    Output shape: [B, num_tokens, feat_dim].
    feat_dim = num_attention_heads * attention_head_dim = 2048 for 2B.
    num_tokens depends on input resolution: (H/16)*(W/16) where 16 = VAE(8x) * patch(2x).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        variant: str = "teacher",
        vae_dir: str | None = None,
        action_regime: str = "null",
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

        if action_regime != "null":
            raise NotImplementedError(
                f"action_regime='{action_regime}' is not supported in Phase 6. "
                "Only 'null' (no action conditioning) is implemented."
            )

        self._checkpoint_dir = checkpoint_dir
        self._variant = variant
        self._input_resolution = input_resolution
        self._timestep = timestep
        self._feat_block_idx = feat_block_idx
        self._output_spatial = output_spatial
        self._dtype_str = dtype

        # --- Transformer ---
        ckpt_path = _find_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            self.transformer = _load_transformer(ckpt_path, dtype)
            cfg = self.transformer.config
            self._feat_dim = cfg.num_attention_heads * cfg.attention_head_dim
            self._num_blocks = len(self.transformer.transformer_blocks)
        else:
            self.transformer = None
            self._feat_dim = _COSMOS_2B_CONFIG["num_attention_heads"] * _COSMOS_2B_CONFIG["attention_head_dim"]
            self._num_blocks = _COSMOS_2B_CONFIG["num_layers"]
            logger.warning(
                "DreamDojoTower: no transformer checkpoint at %s — offline mode. "
                "Download DreamDojo 2B pretrain from nvidia/DreamDojo on HuggingFace, "
                "convert DCP to .pt, place in checkpoint_dir.",
                checkpoint_dir,
            )

        # --- VAE ---
        resolved_vae = _resolve_vae_dir(checkpoint_dir, vae_dir)
        if resolved_vae is not None:
            self.vae = _load_vae(resolved_vae, dtype)
        else:
            self.vae = None
            if self.transformer is not None:
                logger.warning(
                    "DreamDojoTower: Cosmos VAE not found (vae_dir=%s, "
                    "checkpoint_dir/vae=%s). encode() returns dummy features. "
                    "Provide vae_dir or place Cosmos VAE at <checkpoint_dir>/vae/.",
                    vae_dir,
                    os.path.join(checkpoint_dir, "vae"),
                )

        # Scheduler created lazily per-device in _get_scheduler().
        self._scheduler = None
        self._scheduler_device = None

        if self._feat_block_idx >= self._num_blocks:
            raise ValueError(f"feat_block_idx={self._feat_block_idx} >= num_blocks={self._num_blocks}")

        self.freeze()
        logger.info(
            "DreamDojoTower ready (checkpoint=%s, variant=%s, res=%d, block=%d/%d, spatial=%d, feat_dim=%d, online=%s)",
            checkpoint_dir,
            variant,
            input_resolution,
            feat_block_idx,
            self._num_blocks,
            output_spatial,
            self._feat_dim,
            self.online,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def online(self) -> bool:
        """True when both transformer and VAE are loaded (real forward pass)."""
        return self.transformer is not None and self.vae is not None

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def encode(self, images: Tensor) -> Tensor:
        b = images.shape[0]
        device = images.device

        if not self.online:
            num_tokens = self._output_spatial**2
            return torch.zeros(
                b,
                num_tokens,
                self._feat_dim,
                dtype=images.dtype,
                device=device,
            )

        pt_dtype = _DTYPE_MAP[self._dtype_str]

        with torch.inference_mode():
            # Normalize to [-1, 1]
            x = to_neg_one_to_one(images).to(dtype=pt_dtype)

            # Resize to target resolution for consistent token count
            if x.shape[-2] != self._input_resolution or x.shape[-1] != self._input_resolution:
                x = F.interpolate(
                    x,
                    size=(self._input_resolution, self._input_resolution),
                    mode="bilinear",
                    align_corners=False,
                )

            # VAE encode — video VAE expects [B, C, T, H, W]
            x_video = x.unsqueeze(2)  # [B, 3, 1, H, W]
            latents = self.vae.encode(x_video).latent_dist.mode()  # [B, 16, 1, H/8, W/8]

            # Concat zero action channel (in_channels=17 = 16 VAE + 1 action)
            action_ch = latents.new_zeros(b, 1, *latents.shape[2:])
            hidden_states = torch.cat([latents, action_ch], dim=1)  # [B, 17, 1, H/8, W/8]

            # Flow-matching noise at target timestep
            noise = torch.randn_like(hidden_states)
            scheduler = self._get_scheduler(device)
            tau = self._nearest_timestep(scheduler)
            noisy = scheduler.scale_noise(hidden_states, tau.expand(b), noise)

            # Zero text embeddings (null-text conditioning)
            text_embed = noisy.new_zeros(b, 1, _COSMOS_2B_CONFIG["text_embed_dim"])

            # Padding mask (all-ones = fully valid; batch=1 for internal broadcast)
            lat_h, lat_w = latents.shape[-2], latents.shape[-1]
            padding_mask = noisy.new_ones(1, 1, lat_h, lat_w)

            # Hook intermediate block for feature extraction
            feat_holder: dict[str, Tensor] = {}

            def _hook(_module, _input, output):
                feat_holder["feat"] = output.detach()

            block = self.transformer.transformer_blocks[self._feat_block_idx]
            handle = block.register_forward_hook(_hook)
            try:
                self.transformer(
                    hidden_states=noisy,
                    timestep=torch.full(
                        (b,),
                        self._timestep,
                        device=device,
                        dtype=torch.long,
                    ),
                    encoder_hidden_states=text_embed,
                    padding_mask=padding_mask,
                )
            finally:
                handle.remove()

            if "feat" not in feat_holder:
                raise RuntimeError(
                    f"DreamDojo forward hook failed to capture features at block {self._feat_block_idx}."
                )

            feats = feat_holder["feat"]  # [B, T'*H'*W', feat_dim]
            assert feats.ndim == 3, f"Expected 3-D [B, N, C], got {feats.ndim}-D {tuple(feats.shape)}"
            assert feats.shape[-1] == self._feat_dim, f"Expected feat_dim={self._feat_dim}, got {feats.shape[-1]}"
            del noise, noisy, hidden_states, latents
            return feats

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def _get_scheduler(self, device: torch.device):
        """Lazily create and cache the flow-matching scheduler for the given device."""
        if self._scheduler is None or self._scheduler_device != device:
            from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

            self._scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
            self._scheduler.set_timesteps(1000, device=device)
            self._scheduler_device = device
        return self._scheduler

    def _nearest_timestep(self, scheduler) -> Tensor:
        """Find the scheduler timestep closest to self._timestep."""
        timesteps = scheduler.timesteps
        idx = torch.argmin(torch.abs(timesteps - self._timestep))
        return timesteps[idx]


# ---------------------------------------------------------------------------
# Checkpoint & model loading helpers
# ---------------------------------------------------------------------------


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


def _resolve_vae_dir(checkpoint_dir: str, vae_dir: str | None) -> str | None:
    """Find a usable Cosmos VAE directory."""
    if vae_dir is not None:
        if os.path.isdir(vae_dir):
            return vae_dir
        logger.warning("Provided vae_dir not found: %s", vae_dir)
        return None

    candidate = os.path.join(checkpoint_dir, "vae")
    if os.path.isdir(candidate):
        return candidate

    env_dir = os.environ.get("COSMOS_VAE_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    return None


def _load_vae(vae_dir: str, dtype: str):
    """Load the Cosmos VAE from a diffusers-format directory."""
    from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos  # noqa: PLC0415

    pt_dtype = _DTYPE_MAP[dtype]
    logger.info("Loading Cosmos VAE from %s ...", vae_dir)
    vae = AutoencoderKLCosmos.from_pretrained(vae_dir, torch_dtype=pt_dtype)
    vae.eval().requires_grad_(False)  # noqa: FBT003
    param_count = sum(p.numel() for p in vae.parameters())
    logger.info("Cosmos VAE loaded (%d params, %.1fM)", param_count, param_count / 1e6)
    return vae


def _load_transformer(ckpt_path: str, dtype: str):
    """Instantiate CosmosTransformer3DModel and load DreamDojo weights.

    Uses diffusers' built-in Cosmos key conversion to map NVIDIA-native
    key names (net.blocks.*, net.x_embedder.*, etc.) to diffusers format
    (transformer_blocks.*, patch_embed.*, etc.). DreamDojo's extra
    action-conditioning keys are skipped via strict=False.
    """
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers  # noqa: PLC0415
    from diffusers.models import CosmosTransformer3DModel  # noqa: PLC0415

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

    action_keys = [k for k in result.unexpected_keys if any(k.startswith(p) for p in _DREAMDOJO_ACTION_KEY_PREFIXES)]
    other_unexpected = [k for k in result.unexpected_keys if k not in set(action_keys)]

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
