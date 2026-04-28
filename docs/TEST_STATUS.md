# openpi-Vega3D Test Status

Living document tracking which tests have been completed and which are still pending for each phase.
Newest phase appears first.

---

## Phase 3: Adaptive Gated Fusion Integration (2026-04-20)

**Goal**: Wire VEGA-3D generative towers into the policy via Adaptive Gated Fusion (paper Eqs. 6-8). Inference-only scope.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, VAE checkpoint at `ckpts/stable-diffusion-2-1-base/vae`, norm stats symlinked from RLinf.

### Completed Tests (7)

| Test | Validates | Result |
|------|-----------|--------|
| Prereq: VAE tower at `output_spatial=16` | Tower produces `[1, 256, 4]` (16×16 grid) matching PaliGemma native token count | **PASS** |
| Prereq: WAN tower at `output_spatial=16` | Tower produces `[1, 256, 1536]` at 16×16 grid | **PASS** |
| `AdaptiveGatedFusion` module unit test | Correct output shape `[B, N, D]`; `force_gate=1.0` → F_sem exactly; `force_gate=0.0` → F_gen exactly; shape mismatch raises; `force_gate` out-of-range raises | **PASS** |
| Baseline construction (`use_vega3d=False`) | Model builds identically to pre-Phase-3. `spatial_tower=None`, `P_gen=None`, `fusion=None`. | **PASS** |
| VEGA-3D construction (`use_vega3d=True`, VAE) | Model builds with `VAETower(output_spatial=16)`, `P_gen=Linear(4→2048)`, `P_sem=Linear(2048→2048)`, `AdaptiveGatedFusion(2048)`. State dict includes all adapter keys. | **PASS** |
| Full inference with VAE fusion | `policy.infer()` on dummy obs returns actions shape `(100, 23)`. Actions differ from baseline by mean |Δ| = 0.435 (tower signal is non-zero). `force_gate=1.0` path also diverges from baseline by 0.383 due to random-init `P_sem`. | **PASS** |
| **End-to-end rollout with VAE fusion** (2026-04-21) | `bash scripts/run_rollout.sh --use_vega3d --gen_tower vae --gen_tower_ckpt ckpts/stable-diffusion-2-1-base --task_name turning_on_radio --skip_load_task_instance --max_steps 5` — full OmniGibson + policy + receding horizon loop. 5 steps executed, 1 replan, `policy.infer()` = 0.22s, 89 ms/step total, clean `og.shutdown()`, exit code 0. Policy-internal tower forward verified at 16×16 grid during rollout. | **PASS** |

### Follow-ups (deferred to Phase 4)

| Item | Notes |
|------|--------|
| Rollout with WAN fusion | Same but `--gen_tower wan_t2v --gen_tower_ckpt ckpts/Wan2.1-T2V-1.3B --gen_tower_prompt_emb ckpts/wan_prompt_embedding.pt`. Memory may be tight on single GPU (WAN is 1.3B + policy 3.5B). |
| Learned-gate usefulness measurement | Requires Phase 4 training of `P_gen` / `P_sem` / `fusion.gate_proj` on B1K data. Ablation: learned vs `force_gate=1.0` vs baseline. |
| `to_unit_range` compile-friendly rewrite | Current `.item()` call breaks `torch.compile`. Not a correctness issue but a training-throughput issue for Phase 4. |
| **Duplicate tower load with `--gen_tower` + `--use_vega3d`** | The legacy standalone `--gen_tower` smoke path (from Phase 1) loads its own tower instance at default `output_spatial=14` *in addition to* the policy-internal tower at 16. Two forward calls per step, two tower instances in memory. Fix: skip the standalone smoke path when `--use_vega3d` is active, since the policy already uses the tower correctly. |

---

## Code Review Fixes (2026-04-20)

Correctness and API cleanliness pass over Phases 0-2 code. All changes verified with `test_tower.py --offline`.

### Fixes Applied

| Fix | File | Phase |
|-----|------|-------|
| `python3.11` → `python3.10` in transformers-replace error message | `src/openpi/models_pytorch/pi0_pytorch.py:124` | 1a |
| `time.time()` → `time.perf_counter()` for all rollout timing | `scripts/run_rollout.py:557,615` | 1b |
| Added `task_id >= num_tasks` bounds check in `embed_suffix` | `src/openpi/models_pytorch/pi0_pytorch.py:280` | 1c |
| Removed dead `video_contexts=None` param from encoder `forward()` | `towers/vae_online_encoder.py:66`, `towers/wan_t2v_encoder.py:228` | 2a |
| Added `assert latents.ndim == 4` guard in `encode()` for both towers | `towers/vae_tower.py:64`, `towers/wan_tower.py:74` | 2b |
| Made WAN output spatial dim configurable (was hardcoded `14`) | `towers/wan_t2v_encoder.py:42,160,221,237`, `towers/wan_tower.py` | 2c |
| Documented `0.18215` SD VAE scaling constant | `towers/vae_online_encoder.py:51` | 2d |
| Replaced `np.True_` with `True` in B1K image masks | `src/openpi/policies/b1k_policy.py:71,75` | 3a |
| Renamed `--replan_interval` → `--chunk_size` with corrected help text | `scripts/run_rollout.py:240,385,400,520,581` | 3b |

### Verification
```
python scripts/test_tower.py --offline  →  OFFLINE VALIDATION: ALL PASSED (16/16)
grep -r video_contexts src/             →  0 source hits
python scripts/run_rollout.py --help    →  --chunk_size visible
```

---

## Phase 2: Standalone BEHAVIOR rollout

**Environment**: RLinf `.venv-openpi`, Python 3.10.20, Isaac Sim / OmniGibson 3.7.1, CUDA policy device. Validated **2026-04-15** with `turning_on_radio`, `--skip_load_task_instance`, `max_steps=3`, real checkpoint + norm stats.

### Completed Tests (4)

| Test | Validates | Result |
|------|-----------|--------|
| `run_rollout.sh` → full episode | `SimpleEnv` + `load_b1k_policy` + `reset` + `infer` + multiple `env.step` + summary logs | **PASS** |
| Process exit | Shell **EXIT=0** (no post-episode segfault when **`og.shutdown()`** runs from **`SimpleEnv.close()`**) | **PASS** |
| Trace visibility | **`[trace og]`**, **`[trace env]`**, **`[trace 06]`+** appear after Kit start (dedicated **`run_rollout`** logger) | **PASS** |
| Root cause of prior **139** | Episode was completing; failure was Kit teardown without **`shutdown`**, plus INFO traces dropped by root WARNING | Documented in **`docs/CHANGELOG.md`** |

### Follow-ups (optional)

| Item | Notes |
|------|--------|
| Align sim **`device`** with GPU | Rollout **`--device cuda`** is for the policy; YAML may still show **`device=cpu`** for OG—only change if you need GPU dynamics / backend parity. |
| Longer episodes / `load_task_instance` | Same path; re-run without **`--skip_load_task_instance`** when TRO instances are required. |

---

## Environment Setup Validation

**Environment**: Python 3.10.19, PyTorch 2.7.1+cu126 (CUDA: True), all core deps installed. OmniGibson 3.7.1, BDDL 3.7.0, diffusers 0.37.1, easydict, einops all available.

### Completed Tests (5)

| Test | Validates | Result |
|------|-----------|--------|
| `uv sync --python 3.10` | Venv created, 239 packages installed, Python 3.10.19 | PASS |
| OmniGibson + BDDL editable install | `import omnigibson` -> 3.7.1, `import bddl` -> OK | PASS |
| Transformers patch applied | `import transformers` -> 4.53.2, patched gemma/paligemma/siglip | PASS |
| All core ML imports | torch, jax, flax, transformers, safetensors, diffusers, easydict, einops | PASS |
| openpi_vega3d imports | `openpi_vega3d`, `openpi_vega3d.towers`, `TOWER_REGISTRY`, `BaseTower` | PASS |

### Bugs Found and Fixed During Setup (4)

| Bug | Fix | File |
|-----|-----|------|
| `av==14.4.0` fails to build (no manylinux wheel, needs ffmpeg 7) | Added `av>=16.0.0` to `override-dependencies` | `pyproject.toml` |
| `datetime.UTC` not available in Python 3.10 | Changed to `datetime.timezone.utc` | `src/openpi/shared/download.py` |
| `gemma_2b_lora_32` variant missing | Added variant with LoRA rank 32 | `src/openpi/models/gemma.py` |
| `task_id` dropped by `preprocess_observation_pytorch` | Added `task_id` and `proprio_visibility_mask` to `SimpleProcessedObservation` | `src/openpi/models_pytorch/preprocessing_pytorch.py` |

---

## Phase 1: Generative Tower Infrastructure

**Environment**: Python 3.10.19, PyTorch 2.7.1+cu126. All tower deps now installed (`diffusers`, `easydict`, `einops`). No SD2.1 or WAN checkpoints deployed.

### Completed Tests (16)

| Test | Validates | Result |
|------|-----------|--------|
| Syntax validation (19 files) | All tower package files parse without errors (`ast.parse`) | PASS |
| BaseTower ABC contract | `encode`, `feat_dim`, `freeze`, `check_output` methods exist and work | PASS |
| DummyTower end-to-end | Subclass with `encode -> [B, 196, 8]` produces correct shape/stats/frozen | PASS |
| Import graph (vae_tower) | Imports only `base` and `vae_online_encoder` | PASS |
| Import graph (wan_tower) | Imports only `base` and `wan_t2v_encoder` | PASS |
| TOWER_REGISTRY keys | Registry advertises `vae` and `wan_t2v` | PASS |
| Lazy registry -- no eager import | `from openpi_vega3d.towers import TOWER_REGISTRY` succeeds without `diffusers` | PASS |
| Lazy registry -- contains check | `"vae" in TOWER_REGISTRY` and `"wan_t2v" in TOWER_REGISTRY` both True | PASS |
| Lazy registry -- unknown key | `TOWER_REGISTRY["nonexistent"]` raises `KeyError` with available keys | PASS |
| Lazy registry -- deferred import | `TOWER_REGISTRY["vae"]` resolves to `VAETower` class | PASS |
| common.py -- split_frames | Splits [4,3,H,W] into [2,3,H,W] x2; None returns single chunk | PASS |
| common.py -- to_unit_range | [0,1] passthrough; [-1,1] rescale; CLIP-normalized undo | PASS |
| common.py -- to_neg_one_to_one | Maps [0,1] input to [-1,1] output | PASS |
| common.py -- resize_center_crop | 256x256 -> 224x224 and 256x256 -> 480x832 correct shapes | PASS |
| common.py -- temporal_resample | 5->3 interpolation, 5->5 no-op, 1->4 repeat | PASS |
| common.py -- resolve_inference_dtype | `bf16` -> `torch.bfloat16` (CUDA) or `torch.float32` (CPU); `fp32` -> `torch.float32` | PASS |

### Completed Checkpoint Tests (2026-04-20)

All 3 previously-pending tower tests now PASS. Checkpoints downloaded under `ckpts/`:
- `ckpts/stable-diffusion-2-1-base/vae/` (from `Manojb/stable-diffusion-2-1-base` — `stabilityai/stable-diffusion-2-1-base` was no longer accessible)
- `ckpts/Wan2.1-T2V-1.3B/` (from `Wan-AI/Wan2.1-T2V-1.3B`)
- `ckpts/wan_prompt_embedding.pt` (symlink to VEGA-3D's embedding)

| Test | Result | Observed Output |
|------|--------|-----------------|
| VAETower full forward pass | **PASS** | `[1, 196, 4]`, frozen, mean=-0.21, std=0.42 |
| WanT2VTower full forward pass | **PASS** | `[1, 196, 1536]`, frozen, mean=0.01, std=0.92 |
| TOWER_REGISTRY live instantiation | **PASS** | Both tower tests construct via `TOWER_REGISTRY[key](**kwargs)` |

### Incidental Fix During Checkpoint Testing

| Issue | Fix | File |
|-------|-----|------|
| WAN model hard-crashed on `assert FLASH_ATTN_2_AVAILABLE` because flash-attn is not installed | Changed import to use the existing `attention()` wrapper (has a PyTorch `scaled_dot_product_attention` fallback) via alias `from .attention import attention as flash_attention` | `src/openpi_vega3d/towers/wan/modules/model.py:10` |

### Notes on Observed Output

- **WAN feat_dim=1536 (not 1280)**: The forward hook captures the block output *after* the MLP projects back up, which for Wan2.1-T2V-1.3B is 1536, not the base `cfg.dim=1280`. The tower docstring in `wan_tower.py` should be updated to reflect this.

---

## Phase 0: Repository Setup and Baseline Validation

**Environment**: Python 3.10.19, PyTorch 2.7.1+cu126. Sim tests run 2026-04-08 with Isaac Sim from RLinf venv, `joylo` installed (gello helpers), `PYTHONPATH` including RLinf `site-packages` (see `scripts/run_rollout.sh`).

### Completed Tests (17)

| Test | Validates | Result |
|------|-----------|--------|
| Syntax validation (9 files) | All Phase 0 files parse without errors (`ast.parse`) | PASS |
| File layout (11 files) | All expected files exist in correct locations | PASS |
| Import graph | No circular deps; `run_rollout -> openpi_vega3d -> openpi` | PASS |
| ExtractTaskID (present) | `task_index=5` produces `task_id=int32(5)` | PASS |
| ExtractTaskID (absent) | Missing `task_index` passes through unchanged | PASS |
| format_obs_for_policy | Correct keys, shapes, dtypes from obs dict | PASS |
| _validate_and_clip_actions | Normal values unchanged; out-of-range clipped to [-1,1]; NaN/inf replaced with 0 | PASS |
| load_task_description | Fallback to `task_name.replace("_", " ")` + JSONL file lookup | PASS |
| Pi0Config B1K fields | Defaults correct (`num_tasks=0`); B1K override (`num_tasks=50`, `task_embedding_scale=1.5`) | PASS |
| nn.Embedding + fusion | `nn.Embedding(50, 2048)` forward pass; `time_emb + scale * task_emb` changes output | PASS |
| 7-tuple unpacking | Both B1K (`task_id=tensor`) and vanilla (`task_id=None`) _preprocess_observation returns | PASS |
| embed_suffix validation | `num_tasks>0` + task_id present works; `num_tasks>0` + task_id None raises ValueError; `num_tasks=0` skips block | PASS |
| **Full PI0Pytorch construction** | Model builds with 3.5B params, `task_embeddings=Embedding(50, 1024)`, all layers correct | **PASS** |
| **load_b1k_policy() end-to-end** | Checkpoint loads (63s), norm stats loaded (2 keys), Policy object created | **PASS** |
| **Full inference** | `policy.infer()` with dummy obs -> actions shape `(100, 23)`, range `[-1.12, 1.01]`, mean `-0.02` | **PASS** |
| **Transform pipeline** | `B1kInputs` -> `ExtractTaskID` correctly produces `task_id` from `task_index` | **PASS** |
| **SimpleEnv construction + reset + step** | `VectorEnvironment` loads task `turning_on_radio`; after `reset()`, `head_rgb`/`left_wrist_rgb`/`right_wrist_rgb` are `(224,224,3)` `uint8`, `proprio` is `(256,)` `float64`; one `env.step(zeros(23))` completes | **PASS** |

### Superseded (see Phase 2)

| Test | Notes |
|------|--------|
| `run_rollout.py` end-to-end | **2026-04-08** saw **exit 139** after a successful episode or during teardown; **2026-04-15** **PASS** with **`SimpleEnv.close()` → **`og.shutdown()`** and **`run_rollout`** logger fix. Details: **`docs/CHANGELOG.md`** Phase 2. |
