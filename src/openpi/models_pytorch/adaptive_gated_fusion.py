"""Adaptive Gated Fusion (VEGA-3D, arXiv:2603.19235, Eqs. 6-8).

Per-token sigmoid-gated convex combination of generative and semantic features:

    g_i = sigmoid( W_g . Concat( LN(F_gen_i), LN(F_sem_i) ) + b_g )
    F_fused_i = (1 - g_i) * F_gen_i + g_i * F_sem_i

The gate is a scalar in [0, 1] computed independently for each spatial position.
LayerNorm on each stream resolves the scale mismatch between generative and
semantic manifolds. The convex combination (not sum) keeps the fused output in
the same magnitude range as the inputs, preventing signal amplification that
would destabilize downstream attention layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveGatedFusion(nn.Module):
    """Per-token gated fusion of two feature streams at matching shape.

    Both inputs must have shape [B, N, D]. The module is stateless w.r.t.
    spatial arrangement; callers must ensure token i in F_gen corresponds to
    the same spatial location as token i in F_sem.
    """

    def __init__(
        self,
        hidden_size: int,
        force_gate: float | None = None,
        gate_init_bias: float = 4.0,
    ):
        """
        Args:
            hidden_size: Feature dimension of both streams (D_llm).
            force_gate: If not None, overrides the learned gate with this fixed
                value in [0, 1] at every position -- used for inference-time
                ablation (e.g. 1.0 => pure semantic, 0.0 => pure generative).
            gate_init_bias: Initial bias of the gate projection. The gate weight
                is zero-initialized, so at step 0 the gate is the constant
                sigmoid(gate_init_bias) at every position regardless of input.
                The positive default (4.0 -> g ~= 0.982) makes the fused output
                start as ~pure F_sem, i.e. vanilla pi05 behavior, so the
                random-init P_gen does not perturb the pretrained model before
                training has had a chance to adapt it. Gradient still flows to
                the gate weight (dg/dW = g(1-g)*concat != 0), so the generative
                stream ramps in as training progresses.
        """
        super().__init__()
        if force_gate is not None and not 0.0 <= force_gate <= 1.0:
            raise ValueError(f"force_gate must be in [0, 1], got {force_gate}")
        self.hidden_size = hidden_size
        self.force_gate = force_gate
        self.ln_gen = nn.LayerNorm(hidden_size)
        self.ln_sem = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(2 * hidden_size, 1)  # W_g and b_g baked in
        # Bias the gate toward the semantic stream at init (see gate_init_bias).
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

    def forward(self, F_gen: torch.Tensor, F_sem: torch.Tensor) -> torch.Tensor:
        if F_gen.shape != F_sem.shape:
            raise ValueError(
                f"Shape mismatch: F_gen={tuple(F_gen.shape)} F_sem={tuple(F_sem.shape)}"
            )
        if F_gen.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim {self.hidden_size}, got {F_gen.shape[-1]}"
            )

        if self.force_gate is not None:
            g = torch.full(
                (*F_gen.shape[:-1], 1),
                self.force_gate,
                device=F_gen.device,
                dtype=F_gen.dtype,
            )
        else:
            # LayerNorm in fp32 for numerical stability (matches PyTorch default
            # behavior for LayerNorm under autocast), then cast back.
            concat = torch.cat([self.ln_gen(F_gen), self.ln_sem(F_sem)], dim=-1)
            g = torch.sigmoid(self.gate_proj(concat))  # [B, N, 1]

        return (1.0 - g) * F_gen + g * F_sem
