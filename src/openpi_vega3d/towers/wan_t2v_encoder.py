import json
import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import letterbox_content_box, resize_letterbox_pad, resolve_inference_dtype, split_frames, to_neg_one_to_one
from .rollout_tower_log import log_tower
from .wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from .wan.modules.model import WanModel
from .wan.modules.vae import WanVAE
from .wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2VOnlineEncoder(nn.Module):
    """
    Online WAN-T2V feature encoder.
    Input: [N, 3, H, W], output: [N, Cg, 14, 14].
    """

    def __init__(self, config):
        super().__init__()
        self.task = getattr(config, "generative_vision_tower_task", getattr(config, "generative_encoder_task", "t2v-1.3B"))
        self.checkpoint_dir = getattr(config, "generative_vision_tower_checkpoint", getattr(config, "generative_encoder_checkpoint", ""))
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getenv("WAN_T2V_CKPT_DIR", "")
        if not self.checkpoint_dir:
            # Backward-compat fallback.
            self.checkpoint_dir = os.getenv("WAN_VACE_CKPT_DIR", "")
        if not self.checkpoint_dir:
            raise ValueError(
                "Online WAN-T2V tower requires `generative_vision_tower_checkpoint` "
                "or `WAN_T2V_CKPT_DIR`."
            )

        self.size = getattr(config, "generative_vision_tower_size", getattr(config, "generative_encoder_size", "1280*720"))
        self.timestep = int(getattr(config, "generative_vision_tower_timestep", getattr(config, "generative_encoder_timestep", 300)))
        self.shift = float(getattr(config, "generative_vision_tower_shift", getattr(config, "generative_encoder_shift", 5.0)))
        self.feat_block_idx = int(getattr(config, "generative_vision_tower_feat_block_idx", getattr(config, "generative_encoder_feat_block_idx", -1)))
        self.output_spatial = int(getattr(config, "generative_vision_tower_output_spatial", 14))
        # Break-1 fix (Phase 8.1): pool the output grid from the letterbox
        # CONTENT region only, excluding pure-pad (black-bar) tokens, so token
        # i of the pooled grid covers the same image fraction as SigLIP token i.
        # Off by default: existing caches were built with full-canvas pooling
        # and must stay reproducible.
        self.content_region_pool = bool(
            getattr(config, "generative_vision_tower_content_region_pool", False)
        )
        self.prompt_emb_path = str(
            getattr(
                config,
                "generative_vision_tower_prompt_emb_path",
                os.getenv("WAN_PROMPT_EMBED_PATH", os.path.join(os.path.dirname(__file__), "wan_prompt_embedding.pt")),
            )
        )

        if self.task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported WAN task: {self.task}")
        if self.size not in SIZE_CONFIGS:
            raise ValueError(f"Unsupported WAN size: {self.size}")

        self._validate_t2v_checkpoint()

        self.cfg = WAN_CONFIGS[self.task]
        self.param_dtype = resolve_inference_dtype(config)
        self.num_train_timesteps = self.cfg.num_train_timesteps
        self.vae_stride = self.cfg.vae_stride
        self.patch_size = self.cfg.patch_size
        self.frame_width, self.frame_height = SIZE_CONFIGS[self.size]
        self._model_device = torch.device("cpu")

        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.cfg.vae_checkpoint),
            dtype=self.param_dtype,
            device=torch.device("cpu"),
        )
        if not os.path.exists(self.prompt_emb_path):
            raise FileNotFoundError(
                f"Prompt embedding not found: {self.prompt_emb_path}. "
                "Please run scripts/3d/preprocessing/export_wan_prompt_embedding.py first."
            )
        prompt_context = torch.load(self.prompt_emb_path, map_location="cpu", weights_only=True)
        if isinstance(prompt_context, dict):
            if "context" in prompt_context:
                prompt_context = prompt_context["context"]
            elif "embedding" in prompt_context:
                prompt_context = prompt_context["embedding"]
        if not torch.is_tensor(prompt_context):
            raise ValueError(f"Invalid prompt embedding file: {self.prompt_emb_path}")
        self.prompt_context = prompt_context.detach().cpu().contiguous()

        self.model = WanModel.from_pretrained(self.checkpoint_dir).eval().requires_grad_(False)
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._scheduler_device = None

    def _validate_t2v_checkpoint(self):
        cfg_path = os.path.join(self.checkpoint_dir, "config.json")
        if not os.path.exists(cfg_path):
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse WAN config file: {cfg_path}") from exc

        class_name = cfg.get("_class_name", None)
        model_type = cfg.get("model_type", None)
        if class_name is not None and class_name != "WanModel":
            raise ValueError(
                f"WAN-T2V encoder expects `_class_name=WanModel`, got `{class_name}` in {cfg_path}."
            )
        if model_type is not None and model_type != "t2v":
            raise ValueError(
                f"WAN-T2V encoder expects `model_type=t2v`, got `{model_type}` in {cfg_path}."
            )

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert input frames to [-1, 1], then resize to WAN resolution with
        aspect-preserving letterbox padding (no image content is cropped).
        Pad bars are black (-1.0 in [-1, 1]), matching real letterbox video.
        """
        x = to_neg_one_to_one(frames)
        x = resize_letterbox_pad(x, self.frame_height, self.frame_width, pad_value=-1.0)
        return x.to(dtype=self.param_dtype)

    def _slice_content_region(self, feats: torch.Tensor, in_h: int, in_w: int) -> torch.Tensor:
        """Crop [B, C, grid_h, grid_w] token features to the letterbox content
        region (Break-1 fix), so the subsequent adaptive pool never averages
        pure-pad (black-bar) tokens into the output grid.

        The box comes from `letterbox_content_box`, which mirrors
        `resize_letterbox_pad`'s geometry exactly — for LIBERO's square 224^2
        frames on the 832x480 canvas this is the exact 30x30 content square
        (token cols 11:41). Used identically by BOTH forward paths; this is
        not eval-path unification (Break 3, deferred).
        """
        px_h = self.vae_stride[1] * self.patch_size[1]
        px_w = self.vae_stride[2] * self.patch_size[2]
        if px_h != px_w:
            raise ValueError(
                f"content_region_pool assumes square tokens, got {px_h}x{px_w} px/token."
            )
        top, bottom, left, right = letterbox_content_box(
            in_h, in_w, self.frame_height, self.frame_width, px_h
        )
        return feats[:, :, top:bottom, left:right]

    def _get_text_context(self, device: torch.device, batch_size: int):
        context = self.prompt_context.to(device=device, non_blocking=True)
        if torch.is_floating_point(context):
            context = context.to(self.param_dtype)
        return [context] * batch_size

    def _ensure_scheduler_ready(self, device: torch.device):
        if self._scheduler_device != device:
            # Use full train-time schedule to preserve exact timestep semantics.
            self.scheduler.set_timesteps(self.num_train_timesteps, device=device, shift=self.shift)
            self._scheduler_device = device

    def _select_timestep(self, timesteps: torch.Tensor, target_timestep: int) -> torch.Tensor:
        if timesteps.numel() == 0:
            raise ValueError("Scheduler timesteps is empty.")
        tau_tensor = torch.tensor(int(target_timestep), device=timesteps.device, dtype=timesteps.dtype)
        idx = torch.argmin(torch.abs(timesteps - tau_tensor))
        return timesteps[idx]

    def _move_vae_to_device(self, device: torch.device):
        if self.vae.device != device:
            self.vae.model.to(device)
            self.vae.mean = self.vae.mean.to(device=device)
            self.vae.std = self.vae.std.to(device=device)
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
            self.vae.device = device

    def _move_models_to_device(self, device: torch.device):
        if self._model_device != device:
            self.model.to(device)
            self._move_vae_to_device(device)
            self._model_device = device

    def _forward_single_video(self, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, getattr(self.cfg, "dim", 1280), self.output_spatial, self.output_spatial))

        in_h, in_w = int(frames.shape[-2]), int(frames.shape[-1])  # pre-letterbox size, for content_region_pool
        x = self._prepare_frames(frames)
        frame_list = [x[i].unsqueeze(1) for i in range(x.shape[0])]  # [3, 1, H, W]

        self._ensure_scheduler_ready(device)
        tau = self._select_timestep(self.scheduler.timesteps, target_timestep=self.timestep)
        context = self._get_text_context(device=device, batch_size=len(frame_list))

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype):
            base_latents = self.vae.encode(frame_list)
            target_shape = list(base_latents[0].shape)
            seq_len = math.ceil(
                (target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * target_shape[1]
            )

            latent_batch = torch.stack(base_latents, dim=0)
            noise = torch.randn_like(latent_batch)
            noisy_latents = self.scheduler.add_noise(
                original_samples=latent_batch,
                noise=noise,
                timesteps=tau.expand(len(frame_list)),
            )
            noisy_latents_list = [noisy_latents[i] for i in range(len(frame_list))]

            feat_holder = {}

            def _hook(_, __, output):
                feat_holder["feat"] = output.detach()

            block_idx = (len(self.model.blocks) - 1) if self.feat_block_idx < 0 else self.feat_block_idx
            if block_idx < 0 or block_idx >= len(self.model.blocks):
                raise ValueError(f"feat_block_idx out of range: {block_idx}")

            handle = self.model.blocks[block_idx].register_forward_hook(_hook)
            out_batch = None
            try:
                t = tau.expand(len(frame_list)).to(device=device, dtype=torch.long)
                out_list = self.model(
                    noisy_latents_list,
                    t=t,
                    context=context,
                    seq_len=seq_len,
                )
                out_batch = torch.stack(out_list, dim=0)
            finally:
                handle.remove()

            if "feat" not in feat_holder:
                raise RuntimeError("Failed to capture WAN-T2V intermediate features.")

            feats = feat_holder["feat"]  # [N, L, C]
            grid_h = self.frame_height // (self.vae_stride[1] * self.patch_size[1])
            grid_w = self.frame_width // (self.vae_stride[2] * self.patch_size[2])
            tokens_per_frame = grid_h * grid_w
            if feats.shape[1] != tokens_per_frame:
                raise RuntimeError(
                    f"Unexpected token count: {feats.shape[1]} (expected {tokens_per_frame})."
                )

            feats = feats.view(feats.shape[0], grid_h, grid_w, feats.shape[2]).permute(0, 3, 1, 2).contiguous()
            if self.content_region_pool:
                feats = self._slice_content_region(feats, in_h, in_w)
            feats = F.adaptive_avg_pool2d(feats, output_size=(self.output_spatial, self.output_spatial))
            # Release large temporaries early to reduce peak memory.
            del latent_batch, noisy_latents, noisy_latents_list, noise
            if out_batch is not None:
                del out_batch
            return feats

    def _forward_window_batch(
        self,
        clips: torch.Tensor,
        device: torch.device,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        """Multi-frame: bundle T frames as ONE clip per batch item.

        Unlike `_forward_single_video` (which treats each input frame as its own
        T=1 batch item — paper-faithful per-frame extraction), this method
        constructs one batch item with a real temporal axis, so WAN's DiT
        cross-frame attention actually runs *within* the clip. The temporal
        slice that gets returned is the **last latent frame** of f_gen — a
        causal summary of the window ending at the current frame, suitable for
        fusion into the current frame's SigLIP tokens.

        This goes beyond the published VEGA-3D code (which is per-frame); it is
        the path used when the downstream consumer is single-frame (e.g., Pi0.5)
        and the only place to inject temporal/dynamics signal is the encoder.

        Args:
            clips: [B, T, 3, H, W] — B clips, each a temporal window of T raw
                frames (any input range; `_prepare_frames` normalizes).
            device: target device.
            noise_seed: optional int. When set, noise is drawn deterministically
                from this seed (per call) — gives reproducible cached features.

        Returns:
            [B, output_spatial**2, C] — last-temporal-slot features, pooled to
            the encoder's `output_spatial` grid (to match the SigLIP token
            count downstream). C = feat_dim of the hooked DiT block.
        """
        if clips.ndim != 5:
            raise ValueError(f"Expected [B, T, 3, H, W], got {tuple(clips.shape)}")
        b, t, c_in, h_in, w_in = clips.shape
        if c_in != 3:
            raise ValueError(f"Expected 3 channels (dim 2), got {c_in}")
        if b == 0 or t == 0:
            feat_dim = getattr(self.cfg, "dim", 1280)
            return clips.new_zeros((b, self.output_spatial * self.output_spatial, feat_dim))

        self._move_models_to_device(device)

        # Prepare frames: flatten B*T, run the existing per-frame prep (range
        # normalize + letterbox to WAN resolution), then reshape to per-clip
        # [3, T, H', W'] for the VAE.
        flat = clips.reshape(b * t, c_in, h_in, w_in).to(device)
        flat = self._prepare_frames(flat)  # [B*T, 3, H', W'] in [-1, 1], param_dtype
        prepared = flat.reshape(b, t, 3, self.frame_height, self.frame_width)
        prepared = prepared.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H', W']

        self._ensure_scheduler_ready(device)
        tau = self._select_timestep(self.scheduler.timesteps, target_timestep=self.timestep)
        context = self._get_text_context(device=device, batch_size=b)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype):
            # VAE.encode takes a list of [3, T, H', W'] clips → list of [C', T_lat, h, w] latents.
            base_latents = self.vae.encode([prepared[i] for i in range(b)])
            target_shape = list(base_latents[0].shape)  # [C', T_lat, h, w]
            t_lat = target_shape[1]
            seq_len = math.ceil(
                (target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * t_lat
            )

            latent_batch = torch.stack(base_latents, dim=0)  # [B, C', T_lat, h, w]
            if noise_seed is None:
                noise = torch.randn_like(latent_batch)
            else:
                gen = torch.Generator(device=device).manual_seed(int(noise_seed))
                noise = torch.randn(
                    latent_batch.shape, generator=gen, device=device, dtype=latent_batch.dtype
                )
            noisy_latents = self.scheduler.add_noise(
                original_samples=latent_batch,
                noise=noise,
                timesteps=tau.expand(b),
            )
            noisy_latents_list = [noisy_latents[i] for i in range(b)]

            feat_holder = {}

            def _hook(_, __, output):
                feat_holder["feat"] = output.detach()

            block_idx = (len(self.model.blocks) - 1) if self.feat_block_idx < 0 else self.feat_block_idx
            if block_idx < 0 or block_idx >= len(self.model.blocks):
                raise ValueError(f"feat_block_idx out of range: {block_idx}")

            handle = self.model.blocks[block_idx].register_forward_hook(_hook)
            try:
                t_tensor = tau.expand(b).to(device=device, dtype=torch.long)
                self.model(noisy_latents_list, t=t_tensor, context=context, seq_len=seq_len)
            finally:
                handle.remove()

            if "feat" not in feat_holder:
                raise RuntimeError("Failed to capture WAN-T2V intermediate features.")

            feats = feat_holder["feat"]  # [B, T_lat * grid_h * grid_w, C]
            grid_h = self.frame_height // (self.vae_stride[1] * self.patch_size[1])
            grid_w = self.frame_width // (self.vae_stride[2] * self.patch_size[2])
            expected = t_lat * grid_h * grid_w
            if feats.shape[1] != expected:
                raise RuntimeError(
                    f"Unexpected token count: {feats.shape[1]} "
                    f"(expected {expected} = T_lat={t_lat} * {grid_h}*{grid_w})."
                )

            # [B, T_lat, grid_h, grid_w, C] → take last temporal slot.
            feats = feats.view(b, t_lat, grid_h, grid_w, feats.shape[-1])
            last_slot = feats[:, -1]  # [B, grid_h, grid_w, C] — causal summary @ current frame
            # Pool spatially to match SigLIP's 16×16 grid downstream.
            last_slot = last_slot.permute(0, 3, 1, 2).contiguous()  # [B, C, grid_h, grid_w]
            if self.content_region_pool:
                last_slot = self._slice_content_region(last_slot, h_in, w_in)
            pooled = F.adaptive_avg_pool2d(
                last_slot, output_size=(self.output_spatial, self.output_spatial)
            )
            # [B, C, S, S] → [B, S*S, C] to match WanT2VTower.encode's contract.
            pooled = pooled.permute(0, 2, 3, 1).reshape(
                b, self.output_spatial * self.output_spatial, -1
            ).contiguous()

            # Release large temporaries early.
            del latent_batch, noisy_latents, noisy_latents_list, noise
            return pooled

    def forward(
        self,
        frames: torch.Tensor,
        split_sizes: List[int] | None = None,
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, getattr(self.cfg, "dim", 1280), self.output_spatial, self.output_spatial))

        device = frames.device
        self._move_models_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outs = []
        for chunk in chunks:
            outs.append(self._forward_single_video(chunk, device=device))
        out = torch.cat(outs, dim=0)
        # log_tower(
        #     "WanT2VOnlineEncoder forward: in=%s out=%s device=%s",
        #     tuple(frames.shape),
        #     tuple(out.shape),
        #     device,
        # )
        return out
