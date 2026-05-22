import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import adaptive_gated_fusion as _agf
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Task conditioning: task embedding added to flow-matching time conditioning.
        self.num_tasks = config.num_tasks
        self.task_embedding_scale = config.task_embedding_scale
        if self.num_tasks > 0:
            self.task_embeddings = nnx.Embed(self.num_tasks, action_expert_config.width, rngs=rngs)
            logger.info(
                "Task embeddings enabled: %d tasks, scale=%s", self.num_tasks, self.task_embedding_scale
            )

        # Knowledge Insulation & FAST auxiliary token loss.
        self.use_knowledge_insulation = config.use_knowledge_insulation
        self.use_fast_auxiliary = config.use_fast_auxiliary
        if config.use_fast_auxiliary:
            self.fast_token_embedding = nnx.Embed(
                config.fast_vocab_size, paligemma_config.width, rngs=rngs
            )
            self.fast_token_proj = nnx.Linear(
                paligemma_config.width, config.fast_vocab_size, rngs=rngs
            )
            self.fast_loss_weight = config.fast_loss_weight
            logger.info(
                "FAST auxiliary enabled: vocab_size=%d, loss_weight=%s",
                config.fast_vocab_size, config.fast_loss_weight,
            )

        # VEGA-3D Adaptive Gated Fusion (paper Eqs. 6-8). The spatial tower itself
        # is PyTorch-only (diffusers VAE / Wan T2V). The JAX training path consumes
        # precomputed `observation.tower_features` (computed offline). For eval
        # inference, opt-in via `vega3d_live_tower_for_inference` to build the
        # torch tower in-process; it runs on the host through jax.pure_callback
        # when precomputed features are missing for a configured camera.
        self.use_vega3d = config.use_vega3d
        self.spatial_tower = None
        self._tower_feat_dim = 0
        self._tower_num_tokens = 0
        if self.use_vega3d:
            hidden = paligemma_config.width  # D_llm, 2048 for gemma_2b
            feat_dim = config.vega3d_tower_feat_dim
            self.P_gen = nnx.Linear(feat_dim, hidden, rngs=rngs)
            self.P_sem = nnx.Linear(hidden, hidden, rngs=rngs)
            self.fusion = _agf.AdaptiveGatedFusion(
                hidden, force_gate=config.vega3d_force_gate, rngs=rngs
            )
            self._spatial_cameras = tuple(config.vega3d_cameras)
            self._tower_feat_dim = feat_dim
            tower_kwargs = dict(config.vega3d_tower_kwargs or {})
            self._tower_num_tokens = int(tower_kwargs.get("output_spatial", 16)) ** 2
            logger.info(
                "VEGA-3D fusion enabled: tower=%s (feat_dim=%d), cameras=%s, gate=%s",
                config.vega3d_tower_name,
                feat_dim,
                self._spatial_cameras,
                "learned" if config.vega3d_force_gate is None else f"forced={config.vega3d_force_gate}",
            )
            if config.vega3d_live_tower_for_inference:
                from openpi_vega3d.towers import TOWER_REGISTRY
                import torch as _torch

                tower_cls = TOWER_REGISTRY[config.vega3d_tower_name]
                spatial_tower = tower_cls(**tower_kwargs)
                spatial_tower.freeze()
                spatial_tower.eval()
                if _torch.cuda.is_available():
                    spatial_tower = spatial_tower.to("cuda")
                # Bypass nnx.Module.__setattr__ — the torch module is not part
                # of the JAX param tree and must stay opaque to NNX traversal.
                object.__setattr__(self, "spatial_tower", spatial_tower)
                logger.info(
                    "VEGA-3D live torch tower constructed for inference (device=%s)",
                    next(spatial_tower.parameters()).device,
                )
        else:
            self.P_gen = None
            self.P_sem = None
            self.fusion = None
            self._spatial_cameras = ()

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _fuse_camera(
        self, gen_feats: at.Array, semantic_tokens: at.Array
    ) -> at.Array:
        """Apply VEGA-3D Adaptive Gated Fusion for one camera stream.

        Args:
            gen_feats: [B, N, feat_dim] precomputed spatial-tower features.
            semantic_tokens: [B, N, D_llm] PaliGemma image tokens for this camera.

        Returns:
            [B, N, D_llm] fused tokens (same shape as semantic_tokens).
        """
        gen_feats = gen_feats.astype(semantic_tokens.dtype)
        f_gen = self.P_gen(gen_feats)
        f_sem = self.P_sem(semantic_tokens)
        return self.fusion(f_gen, f_sem)

    def _run_torch_tower_host(self, raw_image_nhwc):
        """Host-side trampoline for jax.pure_callback.

        Receives a numpy NHWC float image (materialized JAX array), runs the
        torch tower on GPU if available, returns numpy [B, N, feat_dim] float32.
        """
        import numpy as _np
        import torch as _torch

        if self.spatial_tower is None:
            raise RuntimeError("Live tower called but spatial_tower is None.")
        img_np = _np.asarray(raw_image_nhwc)
        if img_np.ndim != 4 or img_np.shape[-1] != 3:
            raise ValueError(f"Tower expected [B, H, W, 3], got {img_np.shape}")
        img_np = _np.transpose(img_np, (0, 3, 1, 2))  # NHWC -> NCHW
        device = next(self.spatial_tower.parameters()).device
        img_t = _torch.from_numpy(_np.ascontiguousarray(img_np)).to(device)
        with _torch.inference_mode():
            feats = self.spatial_tower.encode(img_t)
        return feats.detach().to(_torch.float32).cpu().numpy()

    def _live_tower_features(
        self, raw_image: at.Array, num_tokens: int
    ) -> at.Array:
        """Run the torch spatial tower from inside a JIT'd JAX call.

        Uses jax.pure_callback so the JIT trace stays valid: the callback
        receives concrete numpy arrays at runtime and returns a JAX array of
        the declared shape.
        """
        batch = raw_image.shape[0]
        out_shape = jax.ShapeDtypeStruct(
            (batch, num_tokens, self._tower_feat_dim), jnp.float32
        )
        return jax.pure_callback(
            self._run_torch_tower_host,
            out_shape,
            raw_image,
            vmap_method="sequential",
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            # VEGA-3D: fuse spatial-tower features into this stream when
            # configured. Features come either from the dataloader (training,
            # precomputed) or, when missing, from a live torch tower invoked
            # through jax.pure_callback (inference-only opt-in via
            # vega3d_live_tower_for_inference).
            if self.use_vega3d and name in self._spatial_cameras:
                gen_feats = None
                if obs.tower_features is not None and name in obs.tower_features:
                    gen_feats = obs.tower_features[name]
                if gen_feats is None:
                    if self.spatial_tower is None:
                        raise ValueError(
                            f"use_vega3d=True but observation.tower_features is missing "
                            f"camera {name!r} and no live tower was constructed "
                            "(set vega3d_live_tower_for_inference=True for eval)."
                        )
                    gen_feats = self._live_tower_features(
                        obs.images[name], image_tokens.shape[1]
                    )
                image_tokens = self._fuse_camera(gen_feats, image_tokens)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # FAST auxiliary tokens (training only, for Knowledge Insulation).
        # Appended at the end of the prefix with causal (autoregressive)
        # masking so the VLM predicts each FAST token from all preceding
        # context. Image/language tokens cannot attend to FAST tokens.
        if self.use_fast_auxiliary and obs.fast_tokens is not None:
            bos = jnp.zeros((obs.fast_tokens.shape[0], 1), dtype=jnp.int32)
            shifted = jnp.concatenate([bos, obs.fast_tokens[:, :-1]], axis=1)
            bos_mask = jnp.ones((obs.fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, obs.fast_token_mask[:, :-1]], axis=1)
            fast_emb = self.fast_token_embedding(shifted)
            tokens.append(fast_emb)
            input_mask.append(shifted_mask)
            ar_mask += [True] * shifted.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.num_tasks > 0:
            if obs.task_id is None:
                raise ValueError("num_tasks > 0 but observation.task_id is None")
            task_emb = self.task_embeddings(obs.task_id).astype(time_emb.dtype)
            time_emb = time_emb + self.task_embedding_scale * task_emb
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # Skip spatial augmentation (RandomCrop / Resize / Rotate) on cameras
        # whose SigLIP tokens get fused with precomputed VEGA-3D features --
        # otherwise per-step random crops/rotations misregister against the
        # cached f_gen and break the token-level gated fusion. ColorJitter
        # still applies (no spatial effect).
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            skip_spatial_aug_cameras=self._spatial_cameras if self.use_vega3d else (),
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        if self.use_knowledge_insulation or self.use_fast_auxiliary:
            # Two-pass forward: prefix → KV cache → suffix.  Required for
            # knowledge insulation (stop-gradient on cache) and/or FAST
            # auxiliary loss (cross-entropy on prefix output).
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
            )

            # FAST auxiliary loss from prefix output.
            fast_loss = None
            fast_len = 0
            if self.use_fast_auxiliary and observation.fast_tokens is not None:
                fast_len = observation.fast_tokens.shape[1]
                fast_start = prefix_tokens.shape[1] - fast_len
                fast_out = prefix_out[:, fast_start:, :]
                fast_logits = self.fast_token_proj(fast_out)
                log_probs = jax.nn.log_softmax(fast_logits, axis=-1)
                target_log_probs = jnp.take_along_axis(
                    log_probs, observation.fast_tokens[:, :, None], axis=-1
                ).squeeze(-1)
                masked_loss = -target_log_probs * observation.fast_token_mask
                num_valid = jnp.maximum(jnp.sum(observation.fast_token_mask, axis=-1), 1)
                fast_loss = jnp.sum(masked_loss, axis=-1) / num_valid

            # Strip FAST tokens from KV cache — the action expert should not
            # attend to auxiliary discrete-action tokens.
            if fast_len > 0:
                cache_k, cache_v = kv_cache
                kv_cache = (cache_k[:, :, :-fast_len, :, :], cache_v[:, :, :-fast_len, :, :])
                prefix_mask = prefix_mask[:, :-fast_len]

            # Knowledge insulation: sever gradient flow from action expert
            # back into the VLM.
            if self.use_knowledge_insulation:
                kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

            # Suffix forward pass with cached prefix.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_for_suffix, suffix_attn_mask], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        else:
            # Original single-pass forward.
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            fast_loss = None

        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if fast_loss is not None:
            action_loss = action_loss + self.fast_loss_weight * fast_loss[:, None]
        return action_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
