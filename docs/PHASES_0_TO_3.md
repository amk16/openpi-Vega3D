# Phases 0–3: How We Got Here

**Last updated:** 2026-05-04
**Status:** All complete. This document is a retrospective reference for anyone continuing the work.

---

## Project Context

**openpi-Vega3D** is a fork of Physical Intelligence's [openpi](https://github.com/physical-intelligence/openpi) that integrates the VEGA-3D paper's generative-tower fusion (arXiv:2603.19235) into the Pi05 policy for the BEHAVIOR-1K robot manipulation benchmark.

The central hypothesis: robot manipulation improves when the policy sees **both** semantic features (from PaliGemma's SigLIP — "what's in the scene") **and** generative features (from a VAE/WAN tower — "what's the geometry/structure"). VEGA-3D fuses these per spatial token via a learned gate.

### Stack

| Layer | Technology |
|-------|-----------|
| Policy model | Pi05 (~3.5B params): PaliGemma vision-language encoder + Gemma action expert |
| Generative towers | SD2.1 VAE (~80M) and WAN T2V 1.3B (from VEGA-3D) |
| Fusion | Adaptive Gated Fusion (paper Eqs. 6-8) — per-token sigmoid-gated convex combination |
| Simulator | Isaac Sim / OmniGibson 3.7.1 (via BEHAVIOR-1K fork) |
| Framework | PyTorch 2.7.1+cu126, Python 3.10 |
| Hardware | NVIDIA RTX 6000 Ada (48GB VRAM) |

---

## Phase 0: Repository Setup and Baseline Validation

**Date:** 2026-04-05 – 2026-04-08
**Goal:** Patch the vanilla openpi clone with B1K support, create the `openpi_vega3d` package, and get a baseline policy running end-to-end.

### What Was Built

The repo started as a vanilla `openpi` clone. Phase 0 added everything needed to load a B1K checkpoint, run inference, and interface with OmniGibson.

**New package: `src/openpi_vega3d/`**

| File | What it does |
|------|-------------|
| `__init__.py` | Package marker. |
| `env.py` | `SimpleEnv` — thin OmniGibson wrapper (~250 lines). Handles obs extraction (head/wrist RGB + proprio), action validation (clamp to [-1,1], NaN removal), TRO instance loading. Replaced RLinf's 1500-line `BehaviorEnv` by removing PPO buffers, reward shaping, crash recovery. |
| `policy_utils.py` | `load_b1k_policy()` — loads a Pi05 model from a safetensors checkpoint and wires up the B1K transform pipeline without needing the full config system. Reads `config.json` from the checkpoint directory for `action_dim`, `paligemma_variant`, etc. |

**New policy transforms: `src/openpi/policies/b1k_policy.py`**

| Transform | What it does |
|-----------|-------------|
| `B1kInputs` | Extracts 23-dim state from 256-dim proprioception via `PROPRIOCEPTION_INDICES["R1Pro"]`, parses camera images to uint8 HWC, builds `image`/`image_mask`/`state` dict. Camera naming: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (PI05 mode). |
| `B1kOutputs` | Truncates model output actions to 23 dims (strips padding). |

**New script: `scripts/run_rollout.py`**

Main entry point for BEHAVIOR rollouts. Uses **receding horizon control**: predict a chunk of 100 actions via `policy.infer()`, execute one at a time, re-plan when the queue empties. Accepts `--use_vega3d` flag (was a placeholder in Phase 0, wired in Phase 3).

### Patches to openpi Core

These modifications were necessary because the B1K checkpoint uses features not present in vanilla openpi:

| File | What changed | Why |
|------|-------------|-----|
| `src/openpi/models/pi0_config.py` | Added 7 fields: `loss_weighting_strategy`, `action_groups`, `group_weights`, `proprio_dropout_dropout_whole_proprio_pct`, `num_tasks`, `task_embedding_scale` | B1K checkpoint has `task_embeddings.weight` (50 tasks). Without `num_tasks` on the config, the model can't create the matching `nn.Embedding` and checkpoint loading fails. |
| `src/openpi/models/model.py` | Added `proprio_visibility_mask` and `task_id` to `Observation` dataclass + `from_dict()` | The denoising loop needs `task_id` to add task-specific conditioning to the time embedding. `Observation` is the carrier. |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 5 changes: (1) create `nn.Embedding` in `__init__`, (2) return 7-tuple from `_preprocess_observation`, (3) inject `task_emb` into `time_emb` in `embed_suffix`, (4-5) thread `task_id` through `forward()` and `sample_actions()`→`denoise_step()` | Task conditioning: the task embedding is added to the flow-matching time embedding so every denoising step knows which of the 50 tasks it's denoising for. |
| `src/openpi/transforms.py` | Added `ExtractTaskID` transform | Bridges naming convention: rollout script uses `task_index` (int), model expects `task_id`. |
| `src/openpi/models_pytorch/preprocessing_pytorch.py` | Added `proprio_visibility_mask` and `task_id` to `SimpleProcessedObservation` | Preprocessing was dropping `task_id`, causing `embed_suffix` to raise `ValueError`. |

### Tests (17 passing)

Config parsing, B1K field defaults, `nn.Embedding` + fusion math, 7-tuple unpacking, `embed_suffix` validation, full `PI0Pytorch` construction (3.5B params), `load_b1k_policy()` end-to-end, full inference (actions shape `(100, 23)`), transform pipeline, and `SimpleEnv` construction + reset + step.

---

## Environment Setup: Two-Venv Infrastructure

**Date:** 2026-04-08
**Goal:** Create install/launch scripts so openpi-Vega3D can run BEHAVIOR rollouts using Isaac Sim / OmniGibson from the existing RLinf setup.

### The Problem

OmniGibson's `isaacsim` Python package lives in RLinf's venv (Python 3.10, `site-packages/isaacsim`), not in openpi-Vega3D's venv. Simply exporting `ISAAC_PATH` doesn't make `import isaacsim` work — the actual Python package must be on `sys.path`.

### The Solution

| File | What it does |
|------|-------------|
| `scripts/setup_env.sh` | Creates `.venv` (Python 3.10), runs `uv sync`, installs OmniGibson and BDDL from BEHAVIOR-1K, applies transformers patch. |
| `scripts/run_rollout.sh` | Rollout launcher. Prepends RLinf's `site-packages` to `PYTHONPATH`, sets all Isaac Sim / OmniGibson env vars, activates venv, execs `run_rollout.py`. |
| `scripts/verify_env.py` | Diagnostic script. Checks Python version, all required/optional imports, env vars, filesystem assets. |

### Patches

| File | What changed | Why |
|------|-------------|-----|
| `pyproject.toml` | `requires-python: >=3.10` (was `>=3.11`), `target-version: py310` | Isaac Sim requires Python 3.10. |
| `pyproject.toml` | Added `av>=16.0.0` to `override-dependencies` | `av==14.4.0` has no manylinux wheel and fails against ffmpeg 6 on Ubuntu 24.04. |
| `src/openpi/shared/download.py` | `datetime.UTC` → `datetime.timezone.utc` | `datetime.UTC` was added in Python 3.11. |
| `src/openpi/models/gemma.py` | Added `gemma_2b_lora_32` variant (LoRA rank 32) | B1K checkpoint's `config.json` specifies this variant. Upstream only had rank 16. |

### Tests (5 passing)

Venv creation (239 packages), OmniGibson + BDDL import, transformers patch, all ML imports, `openpi_vega3d` imports.

---

## Phase 1: Generative Tower Infrastructure

**Date:** 2026-04-08 – 2026-04-20
**Goal:** Create the pluggable tower registry, port VAE + WAN encoders from VEGA-3D, validate that towers produce features of the expected shape.

### Architecture

```
TOWER_REGISTRY (lazy dict)
    ├── "vae"     → VAETower    → wraps VAEOnlineEncoder    → SD2.1 VAE
    └── "wan_t2v" → WanT2VTower → wraps WanT2VOnlineEncoder → WAN 1.3B
                                                               (single denoising step
                                                                feature extraction)

All towers implement BaseTower(nn.Module, ABC):
    encode(images) → [B, tokens, feat_dim]
    feat_dim → int
    freeze() → freezes all params
    check_output() → diagnostic
```

### Files Created

| File | Purpose |
|------|---------|
| `src/openpi_vega3d/towers/__init__.py` | `TOWER_REGISTRY` — lazy dict. Imports deferred until first access. |
| `src/openpi_vega3d/towers/base.py` | `BaseTower(nn.Module, ABC)` — abstract base class. |
| `src/openpi_vega3d/towers/vae_tower.py` | `VAETower` — SD2.1 VAE encoder. Output: `[B, N, 4]` where N = `output_spatial²`. |
| `src/openpi_vega3d/towers/wan_tower.py` | `WanT2VTower` — WAN T2V feature encoder. Output: `[B, N, 1536]`. |
| `src/openpi_vega3d/towers/common.py` | Image preprocessing utilities (from VEGA-3D). |
| `src/openpi_vega3d/towers/vae_online_encoder.py` | `VAEOnlineEncoder` (from VEGA-3D). |
| `src/openpi_vega3d/towers/wan_t2v_encoder.py` | `WanT2VOnlineEncoder` (from VEGA-3D). |
| `src/openpi_vega3d/towers/wan/` | WAN model subpackage (trimmed from VEGA-3D — only files needed by the encoder). |
| `scripts/test_tower.py` | Phase 1 test script. `--offline` validates syntax + contracts. `--tower vae/wan_t2v` runs a full forward pass. |

### Key Decisions

1. **Lazy registry imports.** `TOWER_REGISTRY["vae"]` triggers the import of `diffusers` only when you actually construct a VAE tower, not when you `from openpi_vega3d.towers import TOWER_REGISTRY`. This keeps the base import graph light.

2. **Trimmed WAN subpackage.** Upstream VEGA-3D's `wan/` imports T5, tokenizers, CLIP, VACE. The encoder only needs `WanModel`, `WanVAE`, and `FlowUniPCMultistepScheduler`. Trimming avoided 5+ unnecessary heavy dependencies.

3. **`output_spatial` is configurable.** Both towers accept `output_spatial` to control the spatial grid size of the output. Default is 14 (paper), but Phase 3 changes it to 16 to match PaliGemma's native SigLIP grid.

### Tower Outputs

| Tower | Checkpoint | Output shape | feat_dim | Notes |
|-------|-----------|-------------|----------|-------|
| VAE | `ckpts/stable-diffusion-2-1-base/vae/` | `[B, 196, 4]` at 14×14 | 4 | VAE latent channels |
| WAN T2V 1.3B | `ckpts/Wan2.1-T2V-1.3B/` | `[B, 196, 1536]` at 14×14 | 1536 | Post-MLP residual width (not `cfg.dim=1280`) |

### Incidental Fix

WAN model hard-crashed on `assert FLASH_ATTN_2_AVAILABLE` because flash-attn wasn't installed. Changed import to use the existing `attention()` wrapper which has a PyTorch `scaled_dot_product_attention` fallback.

### Tests (19 passing)

16 offline tests (syntax, ABC contract, import graph, registry, common.py utilities) + 3 checkpoint tests (VAE forward, WAN forward, live registry instantiation).

---

## Phase 2: Standalone BEHAVIOR Rollout

**Date:** 2026-04-15
**Goal:** Make `scripts/run_rollout.py` a trustworthy end-to-end path: OmniGibson loads, Pi05 infers, steps execute, logs explain failures, process exits cleanly.

### The Problem

Prior to Phase 2, `run_rollout.py` appeared to crash (exit code 139) after apparently successful episodes. The investigation revealed two overlapping causes:

1. **Isaac Sim raises root log level to WARNING after Kit starts.** Our `run_rollout` INFO traces (trace markers, timing, action summaries) were silently suppressed — making it look like the script died when it was actually completing normally.

2. **`env.close()` was a no-op.** OmniGibson's `Environment.close()` / `VectorEnvironment.close()` are intentionally empty. Without calling `omnigibson.shutdown()`, Kit's native resources tore down at interpreter exit, producing segfault (139) on healthy runs.

### The Fix

| Change | File | What it does |
|--------|------|-------------|
| Dedicated `run_rollout` logger | `scripts/run_rollout.py` | Own stream/file handlers with `propagate=False`. Survives Kit's root log level override. |
| `SimpleEnv.close()` → `og.shutdown()` | `src/openpi_vega3d/env.py` | If `omnigibson.app` is set, call `omnigibson.shutdown()` (cleanup + `app.close()`). |
| Trace markers at WARNING level | OmniGibson fork | `[trace og]`, `[trace env]`, `[trace 06]+` markers for load, play, observation space, wrapper stages. |

### Result

Full episode on `turning_on_radio`: load, reset, multiple `policy.infer()` + `env.step()`, summary logs, `og.shutdown()`, **exit code 0**. No more spurious 139s.

### Tests (4 passing)

Full episode run, clean exit, trace visibility, root-cause documented.

---

## Phase 3: Adaptive Gated Fusion Integration

**Date:** 2026-04-20 – 2026-04-21
**Goal:** Wire the VEGA-3D generative towers into the policy via Adaptive Gated Fusion (paper Eqs. 6-8). **Inference-only** — training is Phase 4.

### The Fusion Architecture

```
  raw image ──► VAE tower (frozen) ──► T_gen [B, 256, 4]
                                           │
                                           ▼
                                       P_gen: Linear(4→2048)  ← trainable (Phase 4)
                                           │
                                           ▼ F_gen [B, 256, 2048]
                                           │
  PaliGemma (frozen) ──► T_sem [B, 256, 2048]          │
                              │                         │
                              ▼                         │
                          P_sem: Linear(2048→2048)  ← trainable (Phase 4)
                              │                         │
                              ▼ F_sem [B, 256, 2048]    │
                              │                         │
                              └──────────┬──────────────┘
                                         │
                              ┌──────────▼──────────────┐
                              │  AdaptiveGatedFusion     │ ← trainable (Phase 4)
                              │  ln_gen, ln_sem, gate_proj
                              │                          │
                              │  g_i = sigmoid(W_g · [LN(F_gen_i), LN(F_sem_i)] + b_g)
                              │  F_fused_i = (1-g_i)·F_gen_i + g_i·F_sem_i
                              └──────────┬──────────────┘
                                         │
                                         ▼ F_fused [B, 256, 2048]
                              (replaces base-camera tokens in-place)
```

### Files Created

| File | What it does |
|------|-------------|
| `src/openpi/models_pytorch/adaptive_gated_fusion.py` | `AdaptiveGatedFusion` module (~60 lines). Two LayerNorms, one gate projection, sigmoid, convex combination. `force_gate` knob for ablation. |

### Files Modified

| File | What changed |
|------|-------------|
| `src/openpi/models/pi0_config.py` | Added 5 VEGA-3D fields: `use_vega3d`, `vega3d_tower_name`, `vega3d_tower_kwargs`, `vega3d_cameras`, `vega3d_force_gate`. |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `__init__` instantiates tower + `P_gen` + `P_sem` + `AdaptiveGatedFusion`. New `_fuse_camera()` helper. `_preprocess_observation` returns 8-tuple (added `image_names`). `embed_prefix` applies fusion for cameras in `_spatial_cameras`. Both `forward` and `sample_actions` updated. |
| `src/openpi_vega3d/policy_utils.py` | `load_b1k_policy()` gained VEGA-3D args. Auto-injects `output_spatial=16`. Uses `strict=False` for missing adapter keys. Converts adapters to bf16. |
| `scripts/run_rollout.py` | Wired `--use_vega3d`, `--gen_tower`, `--gen_tower_ckpt`, `--force_gate` flags. |

### Key Decisions and Reasoning

1. **Tower grid = 16×16 = 256 tokens (not the paper's 14×14).** PaliGemma's SigLIP uses a 16×16 grid. Its positional embeddings were learned for that layout. We adjust `output_spatial` on the tower side (cheap `adaptive_avg_pool2d`) rather than destructively pooling PaliGemma's tokens.

2. **Fusion replaces base-camera tokens in-place (not appended as a 4th stream).** The paper's Table 5 ablation: Adaptive-Gated-Fusion (63.2) vs Sequence Concat (59.5) on ScanRefer Acc.25. In-place replacement also avoids attention-mask-arithmetic changes.

3. **`strict=False` on checkpoint load when VEGA-3D is enabled.** Pre-Phase-3 checkpoints don't have `P_gen` / `P_sem` / `fusion.*` keys. Standard adapter pattern — these init randomly and are trained in Phase 4.

4. **Head camera only for now, but `_spatial_cameras` is a tuple.** Extending to wrist cameras in the future requires only adding camera key strings — no refactor needed.

5. **bf16 adapters.** PaliGemma tokens are bf16 post-conversion. Adapters must match or matmul fails. Explicit bf16 conversion at load time is cleaner than per-forward casts.

### Known Issues Carried Into Phase 4

| Issue | Impact | Phase 4 fix |
|-------|--------|-------------|
| `to_unit_range` uses `.item()` — breaks `torch.compile` | ~10-30% throughput loss during training | Sub-phase 4.4 |
| `P_sem` is random init — `force_gate=1.0` doesn't reproduce baseline exactly | Test artifact, not a real problem | Training (4.6+) will adapt it |
| Duplicate tower load when `--use_vega3d` + `--gen_tower` both set | Wastes memory at inference | Low priority cleanup |

### End-to-End Validation (2026-04-21)

Full rollout with VEGA-3D fusion: `run_rollout.sh --use_vega3d --gen_tower vae --gen_tower_ckpt ckpts/stable-diffusion-2-1-base --task_name turning_on_radio --max_steps 5`.

Result: 5 steps executed, 1 replan, `policy.infer()` = 0.22s (comparable to baseline), 89 ms/step, clean `og.shutdown()`, **exit code 0**. Tower forward at 16×16 grid confirmed.

### Tests (7 passing)

VAE tower at 16×16, WAN tower at 16×16, `AdaptiveGatedFusion` unit test (shape + force_gate), baseline construction (no VEGA-3D), VEGA-3D construction (with VAE), full inference with fusion, end-to-end rollout with fusion.

---

## Code Review Fixes (2026-04-20)

A correctness and API cleanliness pass was run over Phases 0–2 code. All verified with `test_tower.py --offline`.

| Fix | File |
|-----|------|
| `python3.11` → `python3.10` in error message | `pi0_pytorch.py:124` |
| `time.time()` → `time.perf_counter()` for rollout timing | `run_rollout.py:557,615` |
| Added `task_id >= num_tasks` bounds check | `pi0_pytorch.py:280` |
| Removed dead `video_contexts=None` param from encoder `forward()` | `vae_online_encoder.py:66`, `wan_t2v_encoder.py:228` |
| Added `assert latents.ndim == 4` guard in `encode()` | `vae_tower.py:64`, `wan_tower.py:74` |
| Made WAN output spatial dim configurable (was hardcoded 14) | `wan_t2v_encoder.py`, `wan_tower.py` |
| Documented `0.18215` SD VAE scaling constant | `vae_online_encoder.py:51` |
| Replaced `np.True_` with `True` in B1K image masks | `b1k_policy.py:71,75` |
| Renamed `--replan_interval` → `--chunk_size` with corrected help text | `run_rollout.py` |

---

## Complete File Inventory

All files created or meaningfully modified across Phases 0–3, organized by package.

### `src/openpi_vega3d/` (new package)

| File | Created in | Purpose |
|------|-----------|---------|
| `__init__.py` | Phase 0 | Package marker |
| `env.py` | Phase 0 | `SimpleEnv` OmniGibson wrapper |
| `policy_utils.py` | Phase 0 | `load_b1k_policy()` checkpoint loader |
| `towers/__init__.py` | Phase 1 | `TOWER_REGISTRY` (lazy) |
| `towers/base.py` | Phase 1 | `BaseTower` ABC |
| `towers/vae_tower.py` | Phase 1 | `VAETower` |
| `towers/wan_tower.py` | Phase 1 | `WanT2VTower` |
| `towers/common.py` | Phase 1 | Image preprocessing utilities |
| `towers/vae_online_encoder.py` | Phase 1 | SD2.1 VAE encoder |
| `towers/wan_t2v_encoder.py` | Phase 1 | WAN T2V feature encoder |
| `towers/wan/` | Phase 1 | WAN model subpackage (trimmed) |

### `src/openpi/` (patched upstream)

| File | Modified in | What changed |
|------|-----------|-------------|
| `models/pi0_config.py` | Phase 0, 3 | +7 B1K fields, +5 VEGA-3D fields |
| `models/model.py` | Phase 0 | +`proprio_visibility_mask`, +`task_id` on `Observation` |
| `models/gemma.py` | Env Setup | +`gemma_2b_lora_32` variant |
| `models_pytorch/pi0_pytorch.py` | Phase 0, 3 | Task embeddings, 8-tuple preprocess, fusion in `embed_prefix` |
| `models_pytorch/preprocessing_pytorch.py` | Env Setup | +`task_id`, +`proprio_visibility_mask` passthrough |
| `models_pytorch/adaptive_gated_fusion.py` | Phase 3 | New file — fusion module |
| `policies/b1k_policy.py` | Phase 0 | New file — `B1kInputs` / `B1kOutputs` |
| `transforms.py` | Phase 0 | +`ExtractTaskID` |
| `shared/download.py` | Env Setup | `datetime.UTC` → `datetime.timezone.utc` |

### `scripts/`

| File | Created in | Purpose |
|------|-----------|---------|
| `setup_env.sh` | Env Setup | Venv creation + install |
| `run_rollout.sh` | Env Setup | Rollout launcher with env vars |
| `run_rollout.py` | Phase 0 | Rollout entry point (receding horizon) |
| `verify_env.py` | Env Setup | Import/env diagnostic |
| `test_tower.py` | Phase 1 | Tower test script |

### Root

| File | Modified in | What changed |
|------|-----------|-------------|
| `pyproject.toml` | Env Setup, Phase 1 | Python 3.10, `av>=16.0.0`, `diffusers>=0.30.0`, `easydict>=1.13` |

---

## Test Summary Across All Phases

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 0 | 17 | All PASS |
| Environment Setup | 5 (+4 bugs found and fixed) | All PASS |
| Phase 1 | 19 (16 offline + 3 checkpoint) | All PASS |
| Phase 2 | 4 | All PASS |
| Code Review | 9 fixes verified | All PASS |
| Phase 3 | 7 | All PASS |
| **Total** | **61 tests** | **All PASS** |

---

## Checkpoints and Assets on Disk

| Path | What | Size | Used by |
|------|------|------|---------|
| `/workspace/RLinf/safetensors_ckpts/openpi_05_20251115_050323_9000_tor/` | Pretrained B1K checkpoint | ~7 GB | Phases 2–4 (policy weights) |
| `ckpts/stable-diffusion-2-1-base/` | SD2.1 VAE checkpoint | ~335 MB | Phases 1, 3, 4 (VAE tower) |
| `ckpts/Wan2.1-T2V-1.3B/` | WAN T2V 1.3B checkpoint | ~17 GB | Phase 1 tests, future Phase 5 |
| `ckpts/wan_prompt_embedding.pt` | WAN prompt embedding (symlink) | ~1 MB | WAN tower inference |
| `outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos/norm_stats.json` | Normalization statistics | ~1 KB | All inference and training |

---

## Where to Find More Detail

| Document | What it covers |
|----------|---------------|
| `docs/CHANGELOG.md` | Detailed per-file change log with code snippets and reasoning |
| `docs/TEST_STATUS.md` | Every test result with environment details and follow-ups |
| `docs/PHASE4_INVESTIGATION.md` | Sub-phase 4.0 investigation: GPU budget, data format, memory model, design decisions |
| `docs/PHASE4_PLAN.md` | Phase 4 handoff plan with sub-phase instructions, code snippets, and risk table |
