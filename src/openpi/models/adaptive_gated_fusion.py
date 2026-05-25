"""Adaptive Gated Fusion (VEGA-3D, paper Eqs. 6-8) — JAX/NNX port.

Mirrors `openpi.models_pytorch.adaptive_gated_fusion.AdaptiveGatedFusion`:

    g_i = sigmoid( W_g . Concat( LN(F_gen_i), LN(F_sem_i) ) + b_g )
    F_fused_i = (1 - g_i) * F_gen_i + g_i * F_sem_i

The gate is a scalar in [0, 1] computed independently per spatial position. The
convex combination keeps the fused output in the same magnitude range as the
inputs, preventing signal amplification that would destabilize downstream
attention layers.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class AdaptiveGatedFusion(nnx.Module):
    """Per-token gated fusion of two feature streams at matching shape.

    Both inputs must have shape [B, N, D]. The module is stateless w.r.t.
    spatial arrangement; callers must ensure token i in F_gen corresponds to
    the same spatial location as token i in F_sem.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        force_gate: float | None,
        gate_clamp: float | None = None,
        gate_warmup_steps: int | None = None,
        gate_warmup_start: float = 1.0,
        gate_warmup_target: float = 0.5,
        rngs: nnx.Rngs,
    ):
        if force_gate is not None and not 0.0 <= force_gate <= 1.0:
            raise ValueError(f"force_gate must be in [0, 1], got {force_gate}")
        if gate_clamp is not None and not 0.0 < gate_clamp < 0.5:
            raise ValueError(f"gate_clamp must be in (0, 0.5), got {gate_clamp}")
        self.hidden_size = hidden_size
        self.force_gate = force_gate
        self.gate_clamp = gate_clamp
        self.gate_warmup_steps = gate_warmup_steps
        self.gate_warmup_start = gate_warmup_start
        self.gate_warmup_target = gate_warmup_target
        self.ln_gen = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln_sem = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.gate_proj = nnx.Linear(2 * hidden_size, 1, rngs=rngs)

    def __call__(self, f_gen: jnp.ndarray, f_sem: jnp.ndarray, *, step: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        if f_gen.shape != f_sem.shape:
            raise ValueError(f"Shape mismatch: f_gen={f_gen.shape} f_sem={f_sem.shape}")
        if f_gen.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim {self.hidden_size}, got {f_gen.shape[-1]}")

        if self.force_gate is not None:
            g = jnp.full((*f_gen.shape[:-1], 1), self.force_gate, dtype=f_gen.dtype)
            g_mean = jnp.array(self.force_gate, dtype=jnp.float32)
        else:
            concat = jnp.concatenate([self.ln_gen(f_gen), self.ln_sem(f_sem)], axis=-1)
            logit = self.gate_proj(concat)
            g = jax.nn.sigmoid(logit)
            # Compute mean in float32 to avoid bf16 saturation (sigmoid(>6) rounds to 1.0 in bf16).
            g_f32 = jax.nn.sigmoid(logit.astype(jnp.float32))
            if self.gate_clamp is not None:
                lo = self.gate_clamp
                g = lo + (1.0 - 2.0 * lo) * g
                g_f32 = lo + (1.0 - 2.0 * lo) * g_f32
            if self.gate_warmup_steps is not None and step is not None:
                warmup = self.gate_warmup_steps
                peak_step = warmup / 2.0
                # Phase 1 (0 -> peak): cosine-anneal forced value from start -> 0.5
                t1 = jnp.clip(step / peak_step, 0.0, 1.0)
                forced = self.gate_warmup_start + (self.gate_warmup_target - self.gate_warmup_start) * 0.5 * (1.0 - jnp.cos(jnp.pi * t1))
                # Phase 2 (peak → end): cosine-anneal from forced 0.5 → learned
                t2 = jnp.clip((step - peak_step) / (warmup - peak_step), 0.0, 1.0)
                lerp = 0.5 * (1.0 + jnp.cos(jnp.pi * t2))
                in_phase1 = step < peak_step
                effective_lerp = jnp.where(in_phase1, 1.0, lerp)
                effective_forced = jnp.where(in_phase1, forced, self.gate_warmup_target)
                g = effective_lerp * effective_forced + (1.0 - effective_lerp) * g
                g_f32 = effective_lerp * effective_forced + (1.0 - effective_lerp) * g_f32
            g_mean = jnp.mean(g_f32)

        return (1.0 - g) * f_gen + g * f_sem, g_mean
