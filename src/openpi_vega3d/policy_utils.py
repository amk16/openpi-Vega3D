"""Utility for loading the B1K policy without the full config system.

Constructs the Pi05 model, loads weights from safetensors, and wires up
the same transform pipeline that create_trained_policy would build -- but
without needing LeRobotB1KDataConfig or a TrainConfig entry.
"""

import json
import logging
import os

import safetensors.torch
import torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models import tokenizer as _tokenizer
from openpi.policies.b1k_policy import B1kInputs, B1kOutputs
from openpi.policies.policy import Policy
from openpi.shared import normalize as _normalize
from openpi import transforms as _transforms

logger = logging.getLogger(__name__)


def load_b1k_policy(
    checkpoint_dir: str,
    norm_stats_dir: str,
    *,
    device: str = "cuda",
    action_horizon: int = 100,
    num_tasks: int = 50,
    task_embedding_scale: float = 1.5,
    use_vega3d: bool = False,
    vega3d_tower_name: str = "vae",
    vega3d_tower_kwargs: dict | None = None,
    vega3d_cameras: tuple[str, ...] = ("base_0_rgb",),
    vega3d_force_gate: float | None = None,
) -> Policy:
    """Load a B1K Pi05 policy from checkpoint for inference.

    This bypasses the config system (TrainConfig / DataConfigFactory) and
    directly constructs the model + transforms.  The resulting Policy has
    the same infer() interface as one produced by create_trained_policy.

    Args:
        checkpoint_dir: Path to checkpoint directory containing model.safetensors.
        norm_stats_dir: Path to directory containing norm_stats.json.
        device: PyTorch device string.
        action_horizon: Length of the action chunk predicted per inference call.
        num_tasks: Number of task embeddings in the checkpoint.
        task_embedding_scale: Scale factor for task embedding contribution.
        use_vega3d: If True, enable VEGA-3D adaptive gated fusion on base camera.
        vega3d_tower_name: Tower registry key ("vae" or "wan_t2v").
        vega3d_tower_kwargs: Passed to the tower constructor. If `output_spatial`
            is not set, 16 is injected so the tower matches PaliGemma's 16x16 grid.
        vega3d_cameras: Which image streams to fuse (currently only base tested).
        vega3d_force_gate: Ablation knob. If set, fusion gate is forced to this
            value instead of being computed; 0.0=pure generative, 1.0=pure semantic.
    """
    ckpt_config_path = os.path.join(checkpoint_dir, "config.json")
    ckpt_overrides = {}
    if os.path.exists(ckpt_config_path):
        with open(ckpt_config_path) as f:
            ckpt_overrides = json.load(f)
        logger.info("Loaded checkpoint config: %s", ckpt_overrides)

    # Default tower grid to 16x16 so it aligns with PaliGemma's native 256-token
    # SigLIP output. We only inject when the caller hasn't specified it.
    if use_vega3d:
        tower_kwargs = dict(vega3d_tower_kwargs or {})
        tower_kwargs.setdefault("output_spatial", 16)
    else:
        tower_kwargs = vega3d_tower_kwargs

    config = Pi0Config(
        pi05=True,
        action_dim=ckpt_overrides.get("action_dim", 32),
        action_horizon=action_horizon,
        paligemma_variant=ckpt_overrides.get("paligemma_variant", "gemma_2b_lora_32"),
        action_expert_variant=ckpt_overrides.get("action_expert_variant", "gemma_300m"),
        num_tasks=num_tasks,
        task_embedding_scale=task_embedding_scale,
        use_vega3d=use_vega3d,
        vega3d_tower_name=vega3d_tower_name,
        vega3d_tower_kwargs=tower_kwargs,
        vega3d_cameras=vega3d_cameras,
        vega3d_force_gate=vega3d_force_gate,
    )

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    logger.info("Loading PI0Pytorch model from %s ...", weight_path)
    model = PI0Pytorch(config=config)
    # B1K checkpoints trained before Phase 3 do not contain P_gen / P_sem / fusion
    # weights. Use strict=False so those randomly-initialized adapters remain
    # (we'll be training them in Phase 4); the rest of the weights still load.
    missing, unexpected = safetensors.torch.load_model(
        model, weight_path, strict=not use_vega3d
    )
    if use_vega3d and missing:
        logger.info(
            "VEGA-3D adapters not in checkpoint (expected for Phase 3): %d missing keys",
            len(missing),
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # VEGA-3D adapters (P_gen, P_sem, fusion LayerNorms + gate_proj) init as fp32,
    # but PaliGemma tokens flow through the pipeline in bf16 after the selective
    # conversion above. Match the adapters' dtype so the fused matmul works.
    if use_vega3d:
        model.P_gen.to(dtype=torch.bfloat16)
        model.P_sem.to(dtype=torch.bfloat16)
        model.fusion.to(dtype=torch.bfloat16)
        logger.info("Converted VEGA-3D adapters to bfloat16")

    logger.info("Model loaded successfully")

    logger.info("Loading norm stats from %s ...", norm_stats_dir)
    norm_stats = _normalize.load(norm_stats_dir)
    logger.info("Norm stats loaded (%d keys)", len(norm_stats))

    input_transforms = [
        B1kInputs(action_dim=config.action_dim, model_type=config.model_type),
        _transforms.Normalize(norm_stats, use_quantiles=True),
        _transforms.ResizeImages(224, 224),
        _transforms.ExtractTaskID(),
        _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(config.max_token_len),
            discrete_state_input=config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(config.action_dim),
    ]

    output_transforms = [
        _transforms.Unnormalize(norm_stats, use_quantiles=True),
        B1kOutputs(action_dim=23),
    ]

    return Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        is_pytorch=True,
        pytorch_device=device,
    )
