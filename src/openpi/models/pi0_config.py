import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # B1K loss weighting (training only, safe to ignore at inference)
    loss_weighting_strategy: str = "per_group"
    action_groups: dict[str, tuple[int, int]] | None = None
    group_weights: dict[str, float] | None = None
    proprio_dropout_dropout_whole_proprio_pct: float = 0.0

    # Task conditioning: task embeddings added to flow-matching time conditioning
    num_tasks: int = 0
    task_embedding_scale: float = 1.0

    # VEGA-3D Adaptive Gated Fusion (Phase 3; paper Eqs. 6-8)
    # When use_vega3d=True, base-camera image tokens are replaced by a gated fusion
    # of generative tower features (P_gen(tower(img))) and PaliGemma's own image
    # tokens (P_sem(sem_tokens)). Grid is set to 16x16 to match PaliGemma's native
    # 256-token layout; tower output_spatial is auto-injected as 16 if unspecified.
    use_vega3d: bool = False
    vega3d_tower_name: str = "vae"  # "vae" | "wan_t2v"
    vega3d_tower_kwargs: dict | None = None
    vega3d_cameras: tuple[str, ...] = ("base_0_rgb",)
    # Runtime ablation: force fusion gate to fixed value in [0, 1] (None = learned).
    # 0.0 -> pure generative; 1.0 -> pure semantic.
    vega3d_force_gate: float | None = None
    # Static feat_dim of the precomputed spatial-tower features. The JAX Pi0
    # consumes precomputed `observation.tower_features` (the PyTorch tower runs
    # offline / in the dataloader), so the projection P_gen needs this dim at
    # init time. Auto-derived from `vega3d_tower_name` in __post_init__ when
    # left as None.
    vega3d_tower_feat_dim: int | None = None
    # When True, do NOT instantiate the PyTorch spatial_tower at model
    # construction time. Saves ~3GB RAM during precomputed-feature training
    # runs where observation.tower_features always supplies the features and
    # the live tower forward path is never taken. Must stay False for any
    # eval/inference run that needs to compute features live from images.
    vega3d_skip_tower_construction: bool = False

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.use_vega3d and self.vega3d_tower_feat_dim is None:
            tower_kwargs = self.vega3d_tower_kwargs or {}
            if self.vega3d_tower_name == "vae":
                object.__setattr__(self, "vega3d_tower_feat_dim", 4)
            elif self.vega3d_tower_name == "wan_t2v":
                object.__setattr__(self, "vega3d_tower_feat_dim", tower_kwargs.get("feat_dim", 1280))
            else:
                raise ValueError(
                    f"Cannot auto-derive vega3d_tower_feat_dim for tower {self.vega3d_tower_name!r}; "
                    "set vega3d_tower_feat_dim explicitly."
                )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        task_id_spec = jax.ShapeDtypeStruct([batch_size], jnp.int32) if self.num_tasks > 0 else None

        if self.use_vega3d:
            tower_kwargs = self.vega3d_tower_kwargs or {}
            output_spatial = tower_kwargs.get("output_spatial", 16)
            num_spatial_tokens = output_spatial * output_spatial
            tower_features_spec = {
                cam: jax.ShapeDtypeStruct(
                    [batch_size, num_spatial_tokens, self.vega3d_tower_feat_dim], jnp.float32
                )
                for cam in self.vega3d_cameras
            }
        else:
            tower_features_spec = None

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                task_id=task_id_spec,
                tower_features=tower_features_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
