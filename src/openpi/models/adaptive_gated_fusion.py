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
        gate_init_bias: float = 4.0,
        rngs: nnx.Rngs,
    ):
        if force_gate is not None and not 0.0 <= force_gate <= 1.0:
            raise ValueError(f"force_gate must be in [0, 1], got {force_gate}")
        self.hidden_size = hidden_size
        self.force_gate = force_gate
        self.ln_gen = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln_sem = nnx.LayerNorm(hidden_size, rngs=rngs)
        # W_g and b_g baked into a single 2D->1 projection. The kernel is
        # zero-initialized and the bias set to gate_init_bias, so at step 0 the
        # gate is the constant sigmoid(gate_init_bias) at every position
        # (g ~= 0.982 by default). The fused output then starts as ~pure F_sem
        # -- vanilla pi05 -- so the random-init P_gen does not perturb the
        # pretrained model before training adapts it. Gradient still flows to
        # the kernel (dg/dW = g(1-g)*concat != 0), so the generative stream
        # ramps in as training progresses.
        self.gate_proj = nnx.Linear(
            2 * hidden_size,
            1,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(gate_init_bias),
            rngs=rngs,
        )

    def __call__(self, f_gen: jnp.ndarray, f_sem: jnp.ndarray) -> jnp.ndarray:
        if f_gen.shape != f_sem.shape:
            raise ValueError(f"Shape mismatch: f_gen={f_gen.shape} f_sem={f_sem.shape}")
        if f_gen.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected last dim {self.hidden_size}, got {f_gen.shape[-1]}")

        if self.force_gate is not None:
            g = jnp.full((*f_gen.shape[:-1], 1), self.force_gate, dtype=f_gen.dtype)
        else:
            concat = jnp.concatenate([self.ln_gen(f_gen), self.ln_sem(f_sem)], axis=-1)
            g = jax.nn.sigmoid(self.gate_proj(concat))

        return (1.0 - g) * f_gen + g * f_sem
