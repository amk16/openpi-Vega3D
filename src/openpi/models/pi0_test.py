# ruff: noqa: PLC0415 — heavy imports (torch, training.config, pi0) are
# deferred into the test functions that need them.

import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


# ---------------------------------------------------------------------------
# Phase 8.5 — fidelity-fix tests (docs/PHASE8_PLAN.md). The Phase-8 breaks were
# geometry/scale bugs that type-checked fine, so these check VALUES and
# STRUCTURE, not shapes. All CPU-safe; no checkpoints.
# ---------------------------------------------------------------------------


def _make_fusion(*, blend_normed: bool, force_gate: float | None = None, hidden: int = 32):
    import openpi.models.adaptive_gated_fusion as _agf

    return _agf.AdaptiveGatedFusion(
        hidden,
        force_gate=force_gate,
        blend_normed=blend_normed,
        rngs=nnx.Rngs(0),
    )


def _imbalanced_streams(hidden: int = 32, scale: float = 100.0):
    import jax.numpy as jnp

    k1, k2 = jax.random.split(jax.random.key(42))
    f_gen = jax.random.normal(k1, (2, 6, hidden), dtype=jnp.float32) * scale
    f_sem = jax.random.normal(k2, (2, 6, hidden), dtype=jnp.float32)
    return f_gen, f_sem


def test_blend_normed_scale_and_legacy_identity():
    """Phase 8.5 test 2 (+ the fusion half of test 4): flag-off reproduces the
    legacy formula exactly; flag-on is scale-bounded under x100 imbalance."""
    import jax.numpy as jnp
    import numpy as np

    hidden = 32
    f_gen, f_sem = _imbalanced_streams(hidden)

    # Flag OFF == legacy: output must equal the raw-blend formula computed
    # manually from the module's own params (LN feeds only the gate).
    legacy = _make_fusion(blend_normed=False, hidden=hidden)
    out_legacy, _ = legacy(f_gen, f_sem)
    concat = jnp.concatenate([legacy.ln_gen(f_gen), legacy.ln_sem(f_sem)], axis=-1)
    g = jax.nn.sigmoid(legacy.gate_proj(concat))
    expected = (1.0 - g) * f_gen + g * f_sem
    np.testing.assert_array_equal(np.asarray(out_legacy), np.asarray(expected))

    # The legacy mechanism (the break): the x100-louder stream dominates the
    # output regardless of the gate.
    cos_gen = float(
        jnp.vdot(out_legacy, f_gen) / (jnp.linalg.norm(out_legacy) * jnp.linalg.norm(f_gen))
    )
    assert cos_gen > 0.95, f"sanity: raw blend should be dominated by the loud stream, cos={cos_gen}"

    # Flag ON: blend of LN'd streams -> per-token norms bounded by ~sqrt(D),
    # independent of the input imbalance.
    normed = _make_fusion(blend_normed=True, hidden=hidden)
    out_normed, _ = normed(f_gen, f_sem)
    tok_norms = jnp.linalg.norm(out_normed, axis=-1)
    assert float(tok_norms.max()) < 2.0 * hidden**0.5, (
        f"normed blend must be scale-bounded, max token norm {float(tok_norms.max()):.2f}"
    )

    # Same contract under force_gate (LN must still reach the blend).
    forced = _make_fusion(blend_normed=True, force_gate=0.5, hidden=hidden)
    out_forced, _ = forced(f_gen, f_sem)
    assert float(jnp.linalg.norm(out_forced, axis=-1).max()) < 2.0 * hidden**0.5


def test_p_gen_mlp_shape_and_checkpoint_backfill():
    """Phase 8.5 test 3: mlp2x_gelu P_gen keeps the (B, N, D_llm) contract and
    its nested params are backfilled by the CheckpointWeightLoader regex."""
    import inspect

    import jax.numpy as jnp
    import numpy as np

    import openpi.models.pi0 as _pi0
    import openpi.training.weight_loaders as _weight_loaders

    # Output contract: (B, 256, 2048) from (B, 256, 1536), same as the Linear.
    mlp = _pi0.MLP2xGELU(1536, 2048, rngs=nnx.Rngs(0))
    out = mlp(jnp.zeros((2, 256, 1536), dtype=jnp.float32))
    assert out.shape == (2, 256, 2048)

    # Backfill: a base checkpoint has no P_gen/fusion params; _merge_params
    # must restore the fresh-init nested MLP params via the missing_regex.
    # (Regex literal mirrored from CheckpointWeightLoader.load — the source
    # guard below fails if the loader's regex drifts from what we test.)
    missing_regex = r".*lora.*|(P_gen|P_sem|fusion|fast_token_embedding|fast_token_proj)/.*"
    assert missing_regex in inspect.getsource(_weight_loaders.CheckpointWeightLoader.load), (
        "CheckpointWeightLoader's missing_regex changed — update this test to match"
    )

    ref = {
        "P_gen": {
            "fc1": {"kernel": np.zeros((1536, 2048), np.float32), "bias": np.zeros((2048,), np.float32)},
            "fc2": {"kernel": np.zeros((2048, 2048), np.float32), "bias": np.zeros((2048,), np.float32)},
        },
        "fusion": {"gate_proj": {"kernel": np.zeros((4096, 1), np.float32)}},
        "PaliGemma": {"img": {"kernel": np.zeros((4, 4), np.float32)}},
    }
    loaded = {"PaliGemma": {"img": {"kernel": np.ones((4, 4), np.float32)}}}  # base ckpt: no P_gen
    merged = _weight_loaders._merge_params(loaded, ref, missing_regex=missing_regex)  # noqa: SLF001 — function under test
    assert merged["P_gen"]["fc1"]["kernel"].shape == (1536, 2048)
    assert merged["P_gen"]["fc2"]["kernel"].shape == (2048, 2048)
    assert merged["fusion"]["gate_proj"]["kernel"].shape == (4096, 1)
    assert merged["PaliGemma"]["img"]["kernel"][0, 0] == 1.0  # loaded weights win where present


def test_legacy_regression_flags_off():
    """Phase 8.5 test 4: all three fidelity flags default OFF; the headline
    config is untouched; the fidelityfix config round-trips all three."""
    cfg = _pi0_config.Pi0Config()
    assert cfg.vega3d_blend_normed is False
    assert cfg.vega3d_p_gen_mlp is False

    # Structure: flag off -> P_gen is the legacy single Linear.
    vega_cfg = _pi0_config.Pi0Config(use_vega3d=True, vega3d_tower_name="wan_t2v", vega3d_use_p_sem=False)
    abstract_model = nnx.eval_shape(vega_cfg.create, jax.random.key(0))
    assert isinstance(abstract_model.P_gen, nnx.Linear)
    assert abstract_model.fusion.blend_normed is False

    import openpi.training.config as _train_config

    headline = _train_config.get_config("pi05_libero_fft_wan_precomp_gatewarmup")
    assert headline.model.vega3d_blend_normed is False
    assert headline.model.vega3d_p_gen_mlp is False
    assert "content_region_pool" not in (headline.model.vega3d_tower_kwargs or {})
    assert not headline.data.tower_features_cache_dir.endswith("_cpool")

    fidelityfix = _train_config.get_config("pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix")
    assert fidelityfix.model.vega3d_blend_normed is True
    assert fidelityfix.model.vega3d_p_gen_mlp is True
    assert fidelityfix.model.vega3d_tower_kwargs["content_region_pool"] is True
    assert fidelityfix.data.tower_features_cache_dir.endswith("_cpool")

    # Structure: flags on -> MLP P_gen + normed fusion.
    abstract_ff = nnx.eval_shape(fidelityfix.model.create, jax.random.key(0))
    import openpi.models.pi0 as _pi0

    assert isinstance(abstract_ff.P_gen, _pi0.MLP2xGELU)
    assert abstract_ff.fusion.blend_normed is True


def test_fusion_jax_torch_parity():
    """Phase 8.5 test 5: identical weights + inputs through the JAX and torch
    fusion modules (and P_gen MLPs) give matching outputs, flag on AND off.

    Today the torch twin has zero coverage and framework divergence is a
    demonstrated failure mode in this codebase. Note: nnx.LayerNorm eps is
    1e-6 vs torch's 1e-5 (pre-existing twin divergence) — inputs here have
    variance >> eps so the effect is far below the tolerance.
    """
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")

    import openpi.models.adaptive_gated_fusion as _agf_jax
    import openpi.models.pi0 as _pi0
    import openpi.models_pytorch.adaptive_gated_fusion as _agf_torch

    hidden = 16
    rng = np.random.default_rng(7)
    x_gen = (rng.normal(0, 50.0, (2, 5, hidden))).astype(np.float32)
    x_sem = (rng.normal(0, 1.0, (2, 5, hidden))).astype(np.float32)

    for blend_normed in (False, True):
        jx = _agf_jax.AdaptiveGatedFusion(
            hidden, force_gate=None, blend_normed=blend_normed, rngs=nnx.Rngs(0)
        )
        th = _agf_torch.AdaptiveGatedFusion(hidden, blend_normed=blend_normed)
        with torch.no_grad():
            # LN init (ones/zeros) already matches; copy the gate projection.
            th.gate_proj.weight.copy_(torch.from_numpy(np.asarray(jx.gate_proj.kernel.value).T))
            th.gate_proj.bias.copy_(torch.from_numpy(np.asarray(jx.gate_proj.bias.value)))
            out_t = th(torch.from_numpy(x_gen), torch.from_numpy(x_sem))
        out_j, _ = jx(x_gen, x_sem)
        np.testing.assert_allclose(
            np.asarray(out_j), out_t.numpy(), rtol=1e-4, atol=1e-4,
            err_msg=f"fusion parity failed with blend_normed={blend_normed}",
        )

    # P_gen MLP parity (exact GELU on both sides by construction).
    jm = _pi0.MLP2xGELU(hidden, 24, rngs=nnx.Rngs(1))
    tm = torch.nn.Sequential(torch.nn.Linear(hidden, 24), torch.nn.GELU(), torch.nn.Linear(24, 24))
    with torch.no_grad():
        tm[0].weight.copy_(torch.from_numpy(np.asarray(jm.fc1.kernel.value).T))
        tm[0].bias.copy_(torch.from_numpy(np.asarray(jm.fc1.bias.value)))
        tm[2].weight.copy_(torch.from_numpy(np.asarray(jm.fc2.kernel.value).T))
        tm[2].bias.copy_(torch.from_numpy(np.asarray(jm.fc2.bias.value)))
        out_tm = tm(torch.from_numpy(x_gen))
    out_jm = jm(x_gen)
    np.testing.assert_allclose(np.asarray(out_jm), out_tm.numpy(), rtol=1e-4, atol=1e-4)
