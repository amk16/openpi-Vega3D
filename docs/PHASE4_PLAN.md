# Phase 4: VEGA-3D Adapter Training — Full Plan

**Last updated:** 2026-05-04
**Status:** In progress. Sub-phase 4.1 complete; 4.3–4.8 remaining.

---

## What Phase 4 Is

Phase 4 trains the VEGA-3D fusion adapters so the gated fusion between the generative tower (VAE) and the semantic encoder (PaliGemma) produces better robot actions than either stream alone.

**Phases 0–3 built the infrastructure:** tower registry, environment wrappers, policy loader, and the adaptive gated fusion module. All inference-only. The fusion module is wired in and structurally functional, but the trainable projections (`P_gen`, `P_sem`) and gate (`fusion.gate_proj`) start from random init — they've never seen a gradient. Phase 4 trains them.

### What Gets Trained (and what stays frozen)

**Frozen (~3.5B params):**
- PaliGemma backbone (SigLIP vision + Gemma LM) — pretrained on web-scale image-text
- Gemma action expert (the denoiser) — pretrained on B1K demos
- VAE tower (~80M params) — pretrained on Stable Diffusion 2.1 image data
- Task embeddings (50 tasks) — trained in Phase 0

**Trainable (~4M params for VAE tower):**

| Component | Location | Params | What it learns |
|-----------|----------|--------|----------------|
| `P_gen` | `nn.Linear(4, 2048)` | ~10K | Projects VAE latents into PaliGemma's hidden space. Learns "how to translate VAE features into something Gemma's attention can use." |
| `P_sem` | `nn.Linear(2048, 2048)` | ~4.2M | Re-projects PaliGemma tokens. Learns to emphasize features that complement the generative stream. |
| `ln_gen`, `ln_sem` | `LayerNorm(2048)` × 2 | ~8K | Normalizes each stream before the gate sees them. Learns the right scale/shift for gate input. |
| `gate_proj` | `nn.Linear(4096, 1)` | ~4K | **The intelligence of the fusion.** Per-token, per-image decision: how much to trust the generative vs semantic stream. |

All of these live in `src/openpi/models_pytorch/pi0_pytorch.py` (lines 130–132) and `src/openpi/models_pytorch/adaptive_gated_fusion.py`.

### The Training Signal

Standard flow-matching MSE loss on action chunks (same loss as the original B1K pretraining):

```
L = mean( (u_t − v_t)² )
```

where `u_t` = model's predicted velocity, `v_t` = ground-truth velocity from B1K demos. Gradients flow back through the frozen Gemma expert and PaliGemma backbone to `F_fused`, then through the fusion module to `P_gen`, `P_sem`, `ln_*`, and `gate_proj`.

---

## Locked Decisions

These were established during Sub-phase 4.0 (investigation, documented in `docs/PHASE4_INVESTIGATION.md`) and should not be revisited unless you hit a specific wall:

| Decision | Value | Why |
|----------|-------|-----|
| Training scope | Adapter-only (`P_gen`, `P_sem`, `fusion.*`) | ~4M trainable / ~3.5B frozen. Preserves B1K pretraining; fast convergence. |
| Tower | VAE only (WAN deferred to Phase 5) | Simpler, smaller, faster to iterate. |
| GPU | Single 48GB RTX 6000 Ada, no DDP | ~20GB estimated peak; plenty of headroom. |
| Training data | 190 episodes train, 10 val (matches upstream) | Results directly comparable to baseline. |
| Spatial dropout | `p=0.1` during training | Cheap regularization preventing over-dependence on tower. |
| Eval matrix | 5 tasks × 20 rollouts × 3 methods | Statistically meaningful without being overkill. |
| Learning rate | `1e-4` peak (20× upstream's 5e-6) | Standard adapter-only heuristic: `base_lr × sqrt(total/adapter)`. |
| Batch size | 8 | Single-GPU constraint with headroom. |
| Training steps | 50,000 | Adapter-only converges much faster than full-model training. |

---

## Sub-Phase Status

```
4.0  Investigation          ✅ DONE (2026-04-21)   docs/PHASE4_INVESTIGATION.md
4.1  Data config + TrainConfig  ✅ DONE (2026-05-04)   src/openpi/training/config.py
4.2  Data download          ⏭️  SKIPPED             User provides data manually
4.3  Optimizer filter + logging  ⬚ TODO              scripts/train_pytorch.py
4.4  torch.compile fix      ⬚ TODO              src/openpi_vega3d/towers/common.py
4.5  Spatial dropout         ⬚ TODO              src/openpi/models_pytorch/adaptive_gated_fusion.py
4.6  Smoke training run      ⬚ TODO (needs data)  500 steps, verify nothing explodes
4.7  Convergence training    ⬚ TODO (needs data)  10K–50K steps
4.8  Evaluation              ⬚ TODO (needs ckpt)  300 rollouts, write PHASE4_RESULTS.md
```

### Dependency graph

```
4.3  ──────────────────┐
                       │
4.4 (independent) ─────┤
                       │
4.5 (independent) ─────┤
                       │
          [DATA ARRIVES] ──┤
                       │
                       ▼
                      4.6  (smoke run, 500 steps)
                       │
                       ▼
                      4.7  (convergence, 10K–50K steps)
                       │
                       ▼
                      4.8  (evaluation, 300 rollouts)
```

Sub-phases 4.3, 4.4, and 4.5 can be done in any order, in parallel if desired. All three must be done before 4.6. Data only needs to be on disk for 4.6 onward.

---

## Detailed Instructions Per Sub-Phase

### 4.3 — Adapter-only optimizer filter + gate-stats logging

**File:** `scripts/train_pytorch.py` (632 lines, find the optimizer construction around line 458)

**What to do:**

1. **Filter parameters for AdamW.** Instead of passing `model.parameters()` to the optimizer, filter to only the adapter params:

```python
adapter_params = [
    p for n, p in model.named_parameters()
    if any(k in n for k in ("P_gen", "P_sem", "fusion."))
]
optimizer = torch.optim.AdamW(adapter_params, lr=...)
```

2. **Add a param-count sanity check** immediately after:

```python
n_train = sum(p.numel() for p in adapter_params)
n_total = sum(p.numel() for p in model.parameters())
log.info(f"Trainable: {n_train/1e6:.1f}M / {n_total/1e9:.2f}B ({100*n_train/n_total:.3f}%)")
assert n_train < 20_000_000, "Adapter param count unexpectedly high"
```

Expected: ~4.2M trainable for VAE tower.

3. **Add per-step gate stats logging** inside the training loop, after each forward pass:

```python
with torch.no_grad():
    g_mean = gate_values.mean().item()
    g_std  = gate_values.std().item()
    g_hist = torch.histc(gate_values, bins=10, min=0, max=1)
```

Log `gate/mean`, `gate/std`, and `gate/hist` alongside the existing loss metrics. This is the **primary diagnostic** for whether fusion is learning — see "What to watch for" below.

**How to get `gate_values`:** The `AdaptiveGatedFusion.forward()` method currently returns only `F_fused`. You'll need to also return the gate tensor `g` (or store it as `self._last_gate` for inspection). Modify `adaptive_gated_fusion.py` to expose it.

**Validation:** Run a dry-run with `FakeDataConfig` / dummy data. Confirm param count prints ~4.2M, gate stats are logged, existing loss logging still works.

---

### 4.4 — Fix `to_unit_range` torch.compile fallback

**File:** `src/openpi_vega3d/towers/common.py`

**The problem:** `to_unit_range()` calls `tensor.amin().item()` which forces a graph break under `torch.compile` (`.item()` is a host-device sync).

**Recommended fix (Option A — fixed normalization):** Since our pipeline always feeds `[0, 1]` images to the tower, the auto-detection is unnecessary:

```python
def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return x  # Input is already [0, 1] in our pipeline
```

**Alternative (Option B — compile-friendly detection):**

```python
def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    lo, hi = torch.aminmax(x)
    return torch.where(hi <= 1.0 + 1e-6, x, (x - lo) / (hi - lo).clamp_min(1e-8))
```

Option A is preferred — it makes the tower's behavior deterministic and avoids hidden data-dependent branching.

**Validation:** With `torch._dynamo.config.suppress_errors = False`, a training step should complete without dynamo fallback warnings. Compare step throughput before/after.

---

### 4.5 — Training-time spatial dropout

**File:** `src/openpi/models_pytorch/adaptive_gated_fusion.py`

**What to do:** Add a `dropout_p` parameter to `forward()`. During training, with probability `p`, bypass the gate and return pure `F_sem`:

```python
import random

def forward(self, F_gen: torch.Tensor, F_sem: torch.Tensor, *, dropout_p: float = 0.0) -> torch.Tensor:
    if self.training and dropout_p > 0.0 and random.random() < dropout_p:
        return F_sem
    # ... rest of existing forward unchanged ...
```

Use `random.random()` (CPU-side) instead of `torch.rand().item()` to avoid a device sync.

The caller in `pi0_pytorch.py` (`_fuse_camera` method) needs to pass `dropout_p=0.1` during training:

```python
return self.fusion(F_gen, F_sem, dropout_p=0.1 if self.training else 0.0)
```

**Validation:** Unit test — call `forward()` 1000 times in train mode with `dropout_p=0.1`, confirm ~100 ± 30 return exactly `F_sem`. Confirm eval mode never drops. Confirm `dropout_p=0.0` never drops.

---

### 4.6 — Smoke training run (500 steps)

**Requires:** 4.3 + 4.4 + 4.5 complete, training data on disk.

**Before running:** set `behavior_dataset_root` in `src/openpi/training/config.py` inside the `pi05_b1k_vega3d` entry's `DataConfig` to wherever you placed the B1K demos.

**Command (approximate):**

```bash
cd /workspace/openpi-Vega3D
source .venv/bin/activate
torchrun --nproc_per_node=1 scripts/train_pytorch.py \
    --config pi05_b1k_vega3d \
    --num_train_steps 500 \
    --log_interval 10 \
    --save_interval 100
```

**What to watch for (decision tree):**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Loss decreases, `g_std` rises from ~0 to ~0.05–0.1 | ✅ Healthy | Proceed to 4.7 |
| Loss decreases but `g_std` stays near 0 | Gate not gating yet | Continue to ~1K steps; if still flat, try raising gate-only learning rate |
| Loss flat or increasing | Setup bug (bad data, bad freeze, bad LR) | Stop, debug config/data pipeline |
| `g_mean → 0` or `g_mean → 1` rapidly, `g_std → 0` | **Gate collapse** | Add entropy regularization: `L -= 1e-3 * H(g)` where `H(g) = -(g*log(g) + (1-g)*log(1-g)).mean()`. Restart. |
| OOM | Activation memory exceeded | Reduce batch_size to 4, or verify gradient checkpointing is active |
| Step time > 5s | Compile fallback or data loader bottleneck | Check 4.4 fix is applied; profile data loading |

**Save a checkpoint at step 500.** This becomes the starting point for 4.7.

---

### 4.7 — Convergence training (10K–50K steps)

**Requires:** 4.6 passed (loss decreasing, no collapse).

Resume from the 4.6 checkpoint with the same config. Expected wall time: ~6–18 hours on a single 48GB GPU.

**Monitoring:**
- Train loss every 10 steps (sanity)
- Val loss + gate stats every 500 steps (real signal)
- Checkpoint every 1000 steps
- Save best-by-val-loss checkpoint separately

**Stopping criteria:**
- Val loss plateaus for 3 consecutive evaluations, OR
- Step budget exhausted (50K)

**What "done" looks like:**
- Val loss below the baseline (pre-Phase-3 B1K checkpoint with `use_vega3d=False`)
- Gate distribution healthy: `g_mean ∈ [0.3, 0.7]`, `g_std ∈ [0.05, 0.25]`, histogram has spread

**Hyperparameter tuning levers (if needed):**
- Loss not decreasing fast enough → try `peak_lr=2e-4`
- Loss diverging → try `peak_lr=5e-5`
- Gate collapsing → add entropy regularization (`λ=1e-3`)
- OOM → reduce batch to 4, or increase gradient checkpointing coverage

---

### 4.8 — Evaluation (300 rollouts)

**Requires:** Trained checkpoint from 4.7.

**Three method conditions:**

| Method | Config | What it tests |
|--------|--------|---------------|
| **Baseline** | `use_vega3d=False` | Pre-VEGA-3D performance (the B1K checkpoint as-is) |
| **Force-semantic** | `use_vega3d=True, force_gate=1.0` | Trained P_sem projection but tower output suppressed. Isolates whether P_sem alone helps. |
| **Learned gate** | `use_vega3d=True, force_gate=None` | Full trained fusion — the thing we're trying to validate |

**Five evaluation tasks:**

| Task | Category | Why chosen |
|------|----------|-----------|
| `turning_on_radio` | Easy | Known-good smoke task from Phases 2–3 |
| `picking_up_trash` | Easy | Simple manipulation |
| `putting_away_Halloween_decorations` | Medium | Pick-and-place |
| `carrying_in_groceries` | Medium | Navigation + manipulation |
| `chop_an_onion` | Hard | Tool use / precision |

**Protocol:** 20 rollouts per task per method = 5 × 20 × 3 = **300 rollouts total**.

**Metrics per condition:**
- Success rate (proportion of 20 rollouts completing the task)
- Mean episode length (lower = more efficient)
- Mean reward (if BDDL provides shaped reward)

**Output:** Write `docs/PHASE4_RESULTS.md` with:
- Table per task (3 methods × 3 metrics)
- Aggregate across all tasks
- Gate distribution analysis from the trained model
- The thesis to confirm or reject: **learned gate outperforms both ablations on average**

---

## Key File Reference

| File | Role in Phase 4 |
|------|-----------------|
| `src/openpi/training/config.py` | `pi05_b1k_vega3d` TrainConfig entry + `LeRobotB1KDataConfig` factory (done in 4.1) |
| `scripts/train_pytorch.py` | Training loop. Modify for adapter-only optimizer + gate logging (4.3) |
| `src/openpi_vega3d/towers/common.py` | `to_unit_range()` compile fix (4.4) |
| `src/openpi/models_pytorch/adaptive_gated_fusion.py` | Spatial dropout + gate value exposure (4.5 + 4.3) |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `_fuse_camera()` — passes dropout_p during training (4.5) |
| `scripts/run_rollout.py` | Used for evaluation rollouts (4.8) |
| `src/openpi_vega3d/policy_utils.py` | `load_b1k_policy()` — loads checkpoint with `strict=False` for adapter params |

---

## Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| **Gate collapse** (g → 0 or 1 uniformly) | Medium | Watch `g_mean`/`g_std` from step 1. If collapsing, add entropy reg `L -= 1e-3 * H(g)`. |
| **OOM at batch=8** | Low | Est. 20GB << 48GB. If it happens: batch=4, or offload optimizer state. |
| **LeRobot data schema mismatch** | Low | Verify one sample loads before long run. The `RepackTransform` key mapping is the fragile point. |
| **torch.compile breakage** | Low | `TORCHDYNAMO_DISABLE=1` is always available as escape hatch. Only costs ~10-30% throughput. |
| **Val loss doesn't beat baseline** | Medium | Could mean the VAE tower doesn't carry useful info for these tasks. Try WAN tower (Phase 5) before concluding fusion doesn't help. |

---

## Effort Estimate

| Sub-phase | Active engineering time | Wall clock | Notes |
|-----------|----------------------|-----------|-------|
| 4.3 | 1–2 hours | 1–2 hours | Well-understood pattern |
| 4.4 | <1 hour | <1 hour | Small surgical change |
| 4.5 | <1 hour | <1 hour | Small surgical change |
| 4.6 | 1–4 hours | 1–4 hours | Debugging time depends on what breaks |
| 4.7 | ~0 active | 6–18 hours | Just compute; monitor periodically |
| 4.8 | 4–8 hours | 4–8 hours | Rollouts + analysis + writeup |

**Total:** ~1 day of focused engineering for 4.3+4.4+4.5, then ~1 day for 4.6+4.8 with an overnight 4.7 training run in between.

---

## Where to Find More Detail

- **Full investigation (GPU budget, memory model, data format, design decisions):** `docs/PHASE4_INVESTIGATION.md`
- **What changed in each prior phase:** `docs/CHANGELOG.md`
- **Test results for all phases:** `docs/TEST_STATUS.md`
- **Architecture overview:** `docs/CODEBASE_MAP.md` (if it exists) or the Phase 3 section of `CHANGELOG.md` for the fusion architecture
