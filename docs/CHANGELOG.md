# openpi-Vega3D Changelog

Living document tracking all additions and modifications to the repo, organized by phase.
Newest phase appears first.

---

## Phase 3: Adaptive Gated Fusion Integration (2026-04-20)

**Goal**: Wire the VEGA-3D generative towers (VAE, WAN) into the policy's prefix token stream using the Adaptive Gated Fusion mechanism from the VEGA-3D paper (arXiv:2603.19235, Eqs. 6-8). Scope is inference-only — training-time dropout, gradient flow validation, and usefulness measurements are deferred to Phase 4.

### Fusion method (Paper Eqs. 6-8)

Per-token sigmoid-gated convex combination of projected generative features and projected semantic features. Both streams must share shape `[B, N, D_llm]`; `N = 16×16 = 256` in our integration to match PaliGemma's native SigLIP grid (the paper's 14×14 was their choice; we adapt the tower's `output_spatial` instead of pooling PaliGemma).

```
F_gen = P_gen(tower(img))        # [B, 256, 2048]
F_sem = P_sem(paligemma_tokens)  # [B, 256, 2048]
g_i   = sigmoid(W_g · Concat(LN(F_gen_i), LN(F_sem_i)) + b_g)  # [B, 256, 1]
F_fused_i = (1 - g_i) · F_gen_i + g_i · F_sem_i                # [B, 256, 2048]
```

The fused tokens **replace** PaliGemma's base-camera image tokens in-place — no new prefix stream, no change to `image_masks` tuple arity, no change to attention mask arithmetic.

### Why these design choices (per VEGA-3D ablation, Table 5)

- **Adaptive gated fusion > sequence concat > add > cross-attn > channel concat+MLP > only-generative** on average across 3D scene understanding benchmarks.
- LayerNorm before concat resolves the scale mismatch between generative and semantic manifolds.
- Convex combination (not sum) keeps the fused output in the same magnitude range as the inputs.
- Per-token gate (vs channel-level gate) lets the model choose semantic vs generative source per spatial position.

### Files Created

| File | Purpose |
|------|---------|
| `src/openpi/models_pytorch/adaptive_gated_fusion.py` | New module (~60 lines). Implements `AdaptiveGatedFusion` with LayerNorms, gate projection, and a `force_gate` ablation knob (fixes gate to a constant in [0,1] for runtime comparison of pure-semantic / pure-generative behavior). |

### Files Modified

| File | Change |
|------|--------|
| `src/openpi/models/pi0_config.py` | Added 5 fields: `use_vega3d`, `vega3d_tower_name`, `vega3d_tower_kwargs`, `vega3d_cameras`, `vega3d_force_gate`. |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `__init__` instantiates tower (from `TOWER_REGISTRY`), `P_gen` / `P_sem` (both `nn.Linear(*, 2048)`), and `AdaptiveGatedFusion`. New `_fuse_camera()` helper. `_preprocess_observation` returns 8-tuple adding `image_names`. `embed_prefix` accepts `image_names` kwarg and applies fusion to cameras listed in `self._spatial_cameras`. Both `forward` and `sample_actions` updated to unpack the new tuple field. |
| `src/openpi_vega3d/policy_utils.py` | `load_b1k_policy()` gained `use_vega3d`, `vega3d_tower_name`, `vega3d_tower_kwargs`, `vega3d_cameras`, `vega3d_force_gate` args. **Auto-injects `output_spatial=16`** into tower kwargs so the tower's grid matches PaliGemma. Uses `strict=False` when loading B1K checkpoint if VEGA-3D is enabled (adapter params are not in pre-Phase-3 checkpoints). Converts `P_gen` / `P_sem` / `fusion` to bfloat16 after paligemma conversion so dtypes align. |
| `scripts/run_rollout.py` | Repurposed existing `--use_vega3d` flag to enable policy-side fusion (was previously stub). Reused existing `--gen_tower` / `--gen_tower_ckpt` / `--gen_tower_prompt_emb` flags to select tower and checkpoints. Added `--force_gate` for ablation. |

### Key Decisions and Reasoning

1. **Tower grid = 16×16 = 256 tokens, not the paper's 14×14.** PaliGemma's pretrained SigLIP produces 256 image tokens at its native 16×16 grid; its positional embeddings were learned for that layout. Adjusting our tower's `output_spatial` (a lightweight `adaptive_avg_pool2d` we added in Phase 1) to 16 is free. Pooling PaliGemma's pretrained tokens down to 14×14 would mangle its learned positional structure.

2. **Head camera only for now, but `_spatial_cameras` is a tuple.** Phase 4 can extend to wrist cams by adding camera keys to the tuple — no refactor needed. Encoder signature is structured around a list of cameras internally.

3. **`strict=False` on B1K checkpoint load when VEGA-3D is enabled.** Pre-Phase-3 checkpoints do not contain `P_gen` / `P_sem` / `fusion.*` keys. Standard adapter pattern — those weights initialize randomly for Phase 3 validation and will be trained in Phase 4. Trade-off: loses the safety net that catches misspelled keys.

4. **Fusion replaces base-camera tokens in-place, not appended as a 4th stream.** Earlier draft of this plan proposed sequence concat (separate 4th stream). The paper's Table 5 ablation directly compares these: **Adaptive-Gated-Fusion (63.2)** vs **Sequence Concat (59.5)** on ScanRefer Acc.25. In-place replacement is what the paper does; it also avoids any attention-mask-arithmetic changes.

5. **bf16 adapters.** PaliGemma tokens flow in bf16 after `to_bfloat16_for_selected_params()`. The adapters must match or the matmul fails. Explicitly converting `P_gen` / `P_sem` / `fusion` to bf16 at load time is cleaner than per-forward casts.

### Known Issues and Follow-ups

- **Tower `to_unit_range` breaks `torch.compile`** — uses `tensor.amin().item()` which triggers a dynamo fallback to eager mode for that call. Fine for inference (negligible overhead), but Phase 4 training may want a compile-friendly replacement (fixed normalization constants, or `torch.aminmax` without `.item()`).
- **P_sem is random-init at Phase 3.** Test 3 (`force_gate=1.0` — pure-semantic pathway) produces actions that still differ from the baseline by ~0.38 mean abs because P_sem is a random Linear projection, not an identity. Phase 4 training will adapt P_sem.
- **Dtype on `_fuse_camera` boundary.** Tower output is cast to match `semantic_tokens.dtype`. If future towers or backbones change dtype conventions, this is the spot to watch.
- **Duplicate tower load when `--use_vega3d` + `--gen_tower` are both set.** The pre-existing `--gen_tower` flag (Phase 1) still triggers its own standalone smoke-test tower load at default `output_spatial=14`, independent of the policy-internal tower at 16. Two tower instances, two forward calls per step. Fix: make the standalone path a no-op when `--use_vega3d` is active.

### Loose Ends Closed Post-Phase-3 (2026-04-21)

- **WAN tower docstring corrected.** `wan_tower.py` previously said `feat_dim=1280`; actual measured value is 1536 (post-MLP residual-stream width at the hooked block, not `cfg.dim`). Docstring now documents both numbers and the distinction.
- **End-to-end rollout with VEGA-3D validated.** Full `run_rollout.sh --use_vega3d --gen_tower vae --gen_tower_ckpt ...` completed 5 steps on `turning_on_radio` with clean exit 0. `policy.infer()` = 0.22s (comparable to pre-Phase-3 baseline), 89 ms/step, policy-internal tower forward at 16×16 grid confirmed during the rollout loop. The system-level integration (SimpleEnv + VEGA-3D fusion + receding horizon + og.shutdown) composes cleanly.

---

## Environment Setup: Two-Venv Infrastructure

**Goal**: Create the install and launch scripts so openpi-Vega3D can run BEHAVIOR rollouts using Isaac Sim / OmniGibson from the existing RLinf setup, in a dedicated Python 3.10 venv.

### 2026-04-08 — `isaacsim` on `PYTHONPATH` + sim smoke test

OmniGibson imports the `isaacsim` **Python** package at simulator launch. That package is installed under RLinf’s venv (`site-packages/isaacsim`), not inside openpi-Vega3D’s venv. Exporting `ISAAC_PATH` / `CARB_APP_PATH` / `EXP_PATH` alone does not make `import isaacsim` succeed.

**`scripts/run_rollout.sh`**: Prepend `RLINF_SITE_PACKAGES` (default `/workspace/RLinf/.venv/lib/python3.10/site-packages`) to `PYTHONPATH`, export it, and print it in the launch banner. Override with `RLINF_SITE_PACKAGES_OVERRIDE` if your RLinf venv lives elsewhere.

**`scripts/verify_env.py`**: Inserts that path into `sys.path` and checks `import isaacsim` (warn-only).

**Validation**: `SimpleEnv` for `turning_on_radio` — load, `reset()`, one `step(zeros(23))` — **PASS** (observation shapes/dtypes as expected). If Kit is not shut down explicitly, the process may **segfault on exit** (code 139) during native teardown; **`SimpleEnv.close()`** now calls **`omnigibson.shutdown()`** when the app is running so a full rollout can finish with **exit 0** (see Phase 2 below).

**End-to-end `run_rollout.py`**: With checkpoint and norm stats on disk, **OmniGibson + policy in the same process** is **PASS** as of **2026-04-15** (episode completes; clean exit after `og.shutdown()`). Earlier failures were misread as simulator crashes because (1) Isaac raises the **root** log level to **WARNING** after Kit starts, which hid **`run_rollout`** INFO traces, and (2) **`env.close()`** was a no-op upstream, so Kit tore down at process exit and often produced **139**. Mitigations in `scripts/run_rollout.sh`: correct `PYTHONPATH` order; `USE_TORCH=1` / `USE_TF=0`; `TORCHDYNAMO_DISABLE=1`; import order in `run_rollout.py` per script docstring. Optional `--skip_load_task_instance` for faster debugging.

### Files Created

| File | Purpose |
|------|---------|
| `scripts/setup_env.sh` | Main install script. Creates a `.venv` (Python 3.10), runs `uv sync` + editable install, installs OmniGibson and BDDL from BEHAVIOR-1K, applies the transformers patch, and optionally installs JoyLo/Gello via `--install-joylo`. |
| `scripts/run_rollout.sh` | Rollout launcher. Sets all env vars needed by Isaac Sim / OmniGibson (ISAAC_PATH, OMNIGIBSON_DATA_PATH, CARB_APP_PATH, EXP_PATH, rendering flags), activates the policy venv, and exec's `run_rollout.py`. Mirrors RLinf's `run_minimal_rollout.sh`. |
| `scripts/verify_env.py` | Diagnostic script. Checks Python version, all required/optional imports (torch, jax, openpi, omnigibson, bddl, diffusers, gello), env var presence, and filesystem assets (checkpoint, norm_stats, tokenizer). Outputs PASS/WARN/FAIL checklist. |

### Files Edited

#### `pyproject.toml`

Two changes to support Python 3.10 (required by Isaac Sim / OmniGibson):

```toml
# Before:
requires-python = ">=3.11"

# After:
requires-python = ">=3.10"
```

```toml
# Before:
target-version = "py311"

# After:
target-version = "py310"
```

**Why**: Isaac Sim and OmniGibson's binary packages are built for Python 3.10. The RLinf setup uses Python 3.10 venvs for all BEHAVIOR-related work. Without relaxing this constraint, `uv sync` refuses to create a compatible venv.

Also added `av>=16.0.0` to `override-dependencies` because `av==14.4.0` (pulled by lerobot) has no manylinux wheel and fails to build from source against ffmpeg 6 (Ubuntu 24.04). Version 16+ ships pre-built wheels.

---

#### `src/openpi/shared/download.py`

```python
# Before (Python 3.11+ only):
date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)

# After (Python 3.10 compatible):
date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
```

**Why**: `datetime.UTC` was added in Python 3.11. Since the venv uses Python 3.10, this caused `AttributeError` at import time.

---

#### `src/openpi/models/gemma.py`

Added `gemma_2b_lora_32` variant (LoRA rank 32):

```python
if variant == "gemma_2b_lora_32":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0),
                      "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
    )
```

**Why**: The B1K checkpoint (`config.json`) specifies `paligemma_variant: "gemma_2b_lora_32"`. The upstream openpi only had `gemma_2b_lora` (rank 16). Without this variant, `PI0Pytorch.__init__` raises `ValueError: Unknown variant`.

---

#### `src/openpi/models_pytorch/preprocessing_pytorch.py`

Added `proprio_visibility_mask` and `task_id` to `SimpleProcessedObservation`:

```python
return SimpleProcessedObservation(
    images=out_images,
    image_masks=out_masks,
    state=observation.state,
    proprio_visibility_mask=getattr(observation, "proprio_visibility_mask", None),  # NEW
    task_id=getattr(observation, "task_id", None),                                  # NEW
    tokenized_prompt=observation.tokenized_prompt,
    ...
)
```

**Why**: The preprocessing function was dropping `task_id` from the observation, causing `embed_suffix` to raise `ValueError: Task ID is required for task embeddings` during inference.

---

#### `src/openpi_vega3d/policy_utils.py`

Updated to read `config.json` from the checkpoint directory and use its values for `action_dim`, `paligemma_variant`, and `action_expert_variant` instead of hardcoding them.

**Why**: Different checkpoints may use different model variants and action dimensions. Reading from `config.json` makes the loader robust to checkpoint variations.

---

## Phase 2: Standalone BEHAVIOR rollout (env + policy + teardown)

**Goal**: Make `scripts/run_rollout.py` a trustworthy end-to-end path: OmniGibson loads, Pi05 runs, steps execute, logs explain failures, process exits cleanly.

### 2026-04-15 — Diagnostics, logging, and Kit shutdown

**Trace logging (OmniGibson fork under `BEHAVIOR-1K/OmniGibson/`)**: `rollout_trace` / `_og_rollout_trace` emit at **WARNING** on the `run_rollout` logger (and module loggers) so markers survive Kit’s root log level. Stages tagged include **`Environment` load** (`Ld-*`), **`post_play_load`** (`Epp-*`, including **`Epp10skip`** when no scene graph), **`VectorEnvironment`** (`V*`), and **`RGBLowResWrapper`** (`RGB-*`). **Why**: Pin whether a hang or crash is in scene load, play, observation space, or wrapper—not guess from silence.

**`scripts/run_rollout.py` — `configure_logging`**: The **`run_rollout`** logger gets the same stream/file handlers as root with **`propagate = False`** and the user-selected level. **Why**: After Kit starts, root is often WARNING-only; without this, **`[trace 06]`** onward and phase INFO lines disappear even though Python is healthy.

**`src/openpi_vega3d/env.py` — `SimpleEnv.close()`**: If **`omnigibson.app`** is set, call **`omnigibson.shutdown()`** (cleanup + **`app.close()`**), matching OmniGibson examples. **Why**: **`Environment.close()`** / **`VectorEnvironment.close()`** are intentional no-ops; omitting shutdown left Kit alive until interpreter exit and commonly caused **exit 139** after a successful episode.

**`run_rollout.py`**: **`[trace 16]`** / **`[trace 17]`** bracket **`env.close()`**; **`[trace 17]`** may not run if **`app.close()`** ends the process without returning.

---

## Phase 1: Generative Tower Infrastructure

**Goal**: Create the pluggable generative tower registry, copy and adapt VAE + WAN encoder files from VEGA-3D, and validate that towers can be constructed and produce features of the expected shape.

### Files Created

| File | Purpose |
|------|---------|
| `src/openpi_vega3d/towers/__init__.py` | `TOWER_REGISTRY` -- lazy dict that maps tower names (`"vae"`, `"wan_t2v"`) to tower classes. Imports are deferred until first access to avoid pulling in `diffusers`/`easydict`/`einops` at module load time. |
| `src/openpi_vega3d/towers/base.py` | `BaseTower(nn.Module, ABC)` -- abstract base class defining the tower contract: `encode(images) -> [B, tokens, feat_dim]`, `feat_dim` property, `freeze()`, `check_output()` diagnostic. |
| `src/openpi_vega3d/towers/vae_tower.py` | `VAETower` -- wraps `VAEOnlineEncoder` inside the `BaseTower` interface. Takes `checkpoint_dir` pointing to an SD2.1 base checkpoint. Output: `[B, 196, 4]` (196 = 14x14 spatial grid, 4 = VAE latent channels). |
| `src/openpi_vega3d/towers/wan_tower.py` | `WanT2VTower` -- wraps `WanT2VOnlineEncoder` inside `BaseTower`. Takes `checkpoint_dir` + optional `prompt_emb_path`. Output: `[B, 196, 1280]` for the 1.3B model (1280 = hidden dim). |
| `src/openpi_vega3d/towers/common.py` | Copied verbatim from VEGA-3D. Image preprocessing utilities: `to_unit_range`, `to_neg_one_to_one`, `resize_center_crop`, `temporal_resample`, `split_frames`, `resolve_inference_dtype`. |
| `src/openpi_vega3d/towers/vae_online_encoder.py` | Copied verbatim from VEGA-3D. `VAEOnlineEncoder(nn.Module)` that loads an SD2.1 VAE via `diffusers.AutoencoderKL` and encodes frames to latents. |
| `src/openpi_vega3d/towers/wan_t2v_encoder.py` | Copied verbatim from VEGA-3D. `WanT2VOnlineEncoder(nn.Module)` that runs a single WAN denoising step to extract intermediate features from a frozen WAN T2V model. |
| `src/openpi_vega3d/towers/wan/` | WAN model subpackage, selectively copied from VEGA-3D. Contains only the files needed by `WanT2VOnlineEncoder`: configs (3 model variants), modules (attention, model, vae), and utils (UniPC scheduler). Pipeline classes and unused modules (T5, tokenizers, VACE, clip) are excluded. |
| `scripts/test_tower.py` | Phase 1 test script. `--offline` mode validates syntax (19 files), BaseTower contract, import graph, and registry. `--tower vae/wan_t2v` mode runs a full forward pass with a real checkpoint. |

### Files Edited

#### `src/openpi_vega3d/towers/wan/__init__.py`

Trimmed from the upstream VEGA-3D version:

```python
# UPSTREAM (VEGA-3D):
from . import configs, distributed, modules
from .first_last_frame2video import WanFLF2V
from .image2video import WanI2V
from .text2video import WanT2V
from .vace import WanVace, WanVaceMP

# TRIMMED (openpi-Vega3D):
from . import configs, modules
```

**Why**: The pipeline classes (`WanT2V`, `WanI2V`, etc.) import heavy dependencies (T5 encoders, tokenizers, CLIP) that the encoder doesn't need. The `WanT2VOnlineEncoder` imports directly from `wan.modules.model`, `wan.modules.vae`, and `wan.utils.fm_solvers_unipc`, bypassing this `__init__`. Keeping these imports would cause unnecessary `ImportError`s in environments without the full WAN stack.

---

#### `src/openpi_vega3d/towers/wan/modules/__init__.py`

Trimmed to only re-export the modules used by the encoder:

```python
# UPSTREAM: also imports T5Decoder, T5Encoder, T5EncoderModel, T5Model,
#           HuggingfaceTokenizer, VaceWanModel
# TRIMMED:
from .attention import flash_attention
from .model import WanModel
from .vae import WanVAE
```

**Why**: T5, tokenizers, and VACE model pull in `sentencepiece`, additional HuggingFace models, and VACE-specific logic. None are needed for the T2V feature encoder path.

---

#### `src/openpi_vega3d/towers/wan/utils/__init__.py`

Trimmed to only export the UniPC scheduler:

```python
# UPSTREAM: also imports FlowDPMSolverMultistepScheduler, get_sampling_sigmas,
#           retrieve_timesteps, VaceVideoProcessor
# TRIMMED:
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
```

**Why**: The encoder only uses `FlowUniPCMultistepScheduler` for timestep scheduling. The DPM solver and VACE processor are for full video generation pipelines.

---

#### `pyproject.toml`

Added two new dependencies:

```toml
    "diffusers>=0.30.0",
    "easydict>=1.13",
```

**Why**: `diffusers` provides `AutoencoderKL` (VAE tower), `ConfigMixin`, `ModelMixin`, and scheduler base classes (WAN tower). `easydict` is used by WAN config files (`wan/configs/*.py`). Both are required at runtime when a tower is actually constructed, but not at import time thanks to lazy registration.

---

## Phase 0: Repository Setup and Baseline Validation

**Goal**: Patch the vanilla openpi clone with B1K support, create the `openpi_vega3d` package with a minimal environment wrapper and policy loader, and add a standalone rollout script.

### Files Created

| File | Purpose |
|------|---------|
| `src/openpi_vega3d/__init__.py` | Package marker for the new `openpi_vega3d` module. |
| `src/openpi_vega3d/env.py` | `SimpleEnv` -- thin OmniGibson wrapper for inference-only rollouts. Handles observation extraction (head/wrist RGB + proprio), action validation (clamp to [-1,1], NaN removal), and TRO instance loading. Replaces RLinf's 1500-line `BehaviorEnv` with ~250 lines by removing PPO buffers, reward shaping, crash recovery, and diagnostic logging. |
| `src/openpi_vega3d/policy_utils.py` | `load_b1k_policy()` -- loads a Pi05 model from a safetensors checkpoint and wires up the B1K transform pipeline (B1kInputs, Normalize, ResizeImages, ExtractTaskID, TokenizePrompt, PadStatesAndActions) without needing the full config system (`TrainConfig` / `LeRobotB1KDataConfig`). |
| `src/openpi/policies/b1k_policy.py` | B1K-specific data transforms. `B1kInputs` extracts the 23-dim state from 256-dim proprioception via `PROPRIOCEPTION_INDICES["R1Pro"]`, parses images to uint8 HWC, and builds the `image`/`image_mask`/`state` dict. `B1kOutputs` truncates model output actions to 23 dims. |
| `scripts/run_rollout.py` | Main entry point for BEHAVIOR rollouts. Uses receding horizon control: predicts a chunk of 100 actions via `policy.infer()`, executes them one at a time, re-plans when the queue empties. Accepts `--use_vega3d` flag (placeholder for Phase 4). |

### Files Edited

#### `src/openpi/models/pi0_config.py`

Added 7 fields to `Pi0Config` for B1K checkpoint compatibility:

```python
# B1K loss weighting (training only, safe to ignore at inference)
loss_weighting_strategy: str = "per_group"
action_groups: dict[str, tuple[int, int]] | None = None
group_weights: dict[str, float] | None = None
proprio_dropout_dropout_whole_proprio_pct: float = 0.0

# Task conditioning: task embeddings added to flow-matching time conditioning
num_tasks: int = 0
task_embedding_scale: float = 1.0
```

**Why**: The B1K checkpoints were trained with `num_tasks=50`, meaning the saved weights contain a `task_embeddings.weight` tensor. `PI0Pytorch.__init__` must know about `num_tasks` to create the matching `nn.Embedding` layer before `safetensors.torch.load_model` maps weights to it. Without these fields, checkpoint loading fails with missing-key errors.

---

#### `src/openpi/models/model.py`

Added two optional fields to the `Observation` dataclass (after `state`):

```python
# Visibility mask for proprioception dropout (optional).
proprio_visibility_mask: at.Float[ArrayT, "*b s"] | None = None
# Task ID for task-conditioned models (optional).
task_id: at.Int[ArrayT, "*b"] | None = None
```

Updated `from_dict()` to extract them:

```python
return cls(
    images=data["image"],
    image_masks=data["image_mask"],
    state=data["state"],
    proprio_visibility_mask=data.get("proprio_visibility_mask"),  # NEW
    task_id=data.get("task_id"),                                  # NEW
    tokenized_prompt=data.get("tokenized_prompt"),
    tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
    token_ar_mask=data.get("token_ar_mask"),
    token_loss_mask=data.get("token_loss_mask"),
)
```

**Why**: The denoising loop needs the task ID to add task-specific conditioning to the time embedding. `Observation` is the structured container that carries data from the transform pipeline into the model. Without `task_id` on `Observation`, there's no way to deliver the task index to `embed_suffix`.

---

#### `src/openpi/models_pytorch/pi0_pytorch.py`

**Change 1 -- `__init__` (lines 89-90, 113-115)**: Read task config and create embedding layer.

```python
self.num_tasks = getattr(config, "num_tasks", 0)
self.task_embedding_scale = getattr(config, "task_embedding_scale", 1.0)
# ... (existing code) ...
if self.num_tasks > 0:
    logging.info(f"Task embeddings enabled: {self.num_tasks} tasks, scale={self.task_embedding_scale}")
    self.task_embeddings = nn.Embedding(self.num_tasks, action_expert_config.width)
```

**Why**: Creates the `nn.Embedding` layer that the checkpoint weights map to. `getattr` with default `0` ensures vanilla configs still work without task embeddings.

**Change 2 -- `_preprocess_observation` (lines 168-179)**: Return 7-tuple instead of 5-tuple.

```python
def _preprocess_observation(self, observation, *, train=True):
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
    return (
        list(observation.images.values()),
        list(observation.image_masks.values()),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.state,
        getattr(observation, "proprio_visibility_mask", None),  # NEW
        getattr(observation, "task_id", None),                  # NEW
    )
```

**Why**: Single extraction point called by `forward()`, `sample_actions()`, etc. Extracting `task_id` here means every downstream method gets access without duplicating extraction logic.

**Change 3 -- `embed_suffix` (line 246, lines 278-282)**: Accept `task_id`, inject task embedding into time conditioning.

```python
def embed_suffix(self, state, noisy_actions, timestep, proprio_visibility_mask=None, task_id=None):
    # ... (existing timestep embedding code) ...
    if self.num_tasks > 0:
        if task_id is None:
            raise ValueError("Task ID is required for task embeddings")
        task_emb = self.task_embeddings(task_id)
        time_emb = time_emb + self.task_embedding_scale * task_emb
```

**Why**: This is the core of task conditioning. By adding the task embedding to `time_emb`, every denoising step knows which task it's denoising for. Placed in `embed_suffix` (not `embed_prefix`) because task conditioning needs to affect the Expert Gemma denoiser, not the PaliGemma visual-language encoder.

**Change 4 -- `forward` (line 331, 333, 346)**: Unpack 7-tuple, pass `task_id` to `embed_suffix`.

```python
images, img_masks, lang_tokens, lang_masks, state, proprio_visibility_mask, task_id = self._preprocess_observation(observation, train=True)
# ...
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time, proprio_visibility_mask, task_id)
```

**Why**: Training path must thread `task_id` through to `embed_suffix` for the task embedding addition to execute.

**Change 5 -- `sample_actions` (line 398, 429-430) + `denoise_step` (lines 438-449)**: Thread `task_id` through the inference denoising loop.

```python
# In sample_actions:
images, img_masks, lang_tokens, lang_masks, state, proprio_visibility_mask, task_id = self._preprocess_observation(observation, train=False)
# ...
v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time, proprio_visibility_mask, task_id)

# denoise_step signature:
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep, proprio_visibility_mask=None, task_id=None):
```

**Why**: Inference runs 10 Euler denoising steps. Each step calls `denoise_step` -> `embed_suffix`. Without threading `task_id` all the way down, inference on a B1K checkpoint would crash at `embed_suffix` line 280 (`task_id is None` but `num_tasks > 0`).

---

#### `src/openpi/transforms.py`

Added `ExtractTaskID` transform (inserted before `PadStatesAndActions`):

```python
@dataclasses.dataclass(frozen=True)
class ExtractTaskID(DataTransformFn):
    """Extracts task_id from task_index for task embeddings."""

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" in data:
            return {**data, "task_id": np.int32(data["task_index"])}
        return data
```

**Why**: Bridges two naming conventions. The rollout script and B1K data pipeline use `task_index` (an integer like `0` for `turning_on_radio`). But `Observation.from_dict()` looks for `task_id`. This transform copies the value under the new key name so the model receives it. Without it, `Observation.task_id` would always be `None`, causing `embed_suffix` to raise `ValueError`.
