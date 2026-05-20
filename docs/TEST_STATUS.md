# openpi-Vega3D Test Status

Living document tracking which tests have been completed and which are still pending for each phase.
Newest phase appears first.

---

## Phase 7: DreamDojo Training Setup

### Sub-Phase 7.0 — Merge origin/main (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| Merge completes with single conflict | Clean merge, one conflict in `policy_utils.py` resolved | **PASS** |
| No remaining conflict markers | `grep -rn "<<<<<<" src/ scripts/ docs/` returns empty | **PASS** |
| Key files from main present | `precompute_tower_features.py`, `probe_wan.py`, `train.py` exist | **PASS** |

### Sub-Phase 7.1 — Pi0Config feat_dim (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| `Pi0Config` with `vega3d_tower_name="dreamdojo"` auto-derives `feat_dim=2048` | No explicit `vega3d_tower_feat_dim` needed | **PASS** (code review) |
| `Pi0Config` with unknown tower name still raises `ValueError` | Fallback error preserved | **PASS** (code review) |

### Sub-Phase 7.2 — Probe script (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| `scripts/probe_dreamdojo.py` exists and is syntactically valid | `python3 -c "import ast; ast.parse(open('scripts/probe_dreamdojo.py').read())"` | **PASS** |
| Script prints guidance when no checkpoint present | Offline mode with user-friendly message | **PASS** (code review) |

### Sub-Phase 7.3 — Precompute adaptation (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| `ensure_dreamdojo_checkpoint()` raises `FileNotFoundError` with guidance | Clear download instructions in error message | **PASS** (code review) |
| `prepare_image()` accepts resolution parameter | Default 224 (backward compat), DreamDojo uses 256 | **PASS** (code review) |
| `image_resolution` resolved from tower kwargs | `input_resolution` key read, defaults to 224 | **PASS** (code review) |

### Sub-Phase 7.4 — LIBERO training configs (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| `pi05_libero_lora_dreamdojo` config exists | In-process tower config with batch=4 | **PASS** (code review) |
| `pi05_libero_lora_dreamdojo_precomp` config exists | Precomputed features, skip_tower_construction=True | **PASS** (code review) |
| No DreamDojo semonly config | WAN semonly serves as shared control (force_gate=1.0 zeros generative features regardless of tower) | **PASS** (verified removed) |
| Both configs mirror WAN hyperparameters | 30K steps, 1e-5 LR, cosine decay, same freeze filter | **PASS** (code review) |

### Sub-Phase 7.3b — Multi-frame encode_window_batch (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| `encode()` single-frame regression | Output shape `(1, 256, 2048)` unchanged | **PASS** (real checkpoint) |
| `encode_window_batch(T=17)` | Multi-frame output `(1, 256, 2048)` matches single-frame contract | **PASS** (real checkpoint) |
| `encode_window_batch(T=1)` fallback | Falls back to single-frame `encode()` | **PASS** (real checkpoint) |
| Padding mask shape | `[1, 1, H, W]` — Cosmos repeats along T internally | **PASS** (verified via forward pass) |
| Precomp config updated | `tower_window=17, tower_stride=2, cache_dir=...w17s2...` | **PASS** (code review) |

### Sub-Phase 7.5 — B1K config fix (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| B1K DreamDojo configs (`pi05_b1k_dreamdojo`, `pi05_b1k_dreamdojo_wrist`) commented out | Referenced `LeRobotB1KDataConfig` which is disabled — caused `NameError` at import time | **PASS** (fixed) |
| Re-enable instructions present in comment block | Search "DISABLED: LeRobotB1KDataConfig" in `config.py` | **PASS** |
| No uncommented references to `LeRobotB1KDataConfig` remain | `grep -v '#' config.py \| grep LeRobotB1KDataConfig` returns empty | **PASS** |

### Sub-Phase 7.7 — Base Cosmos control backbone (2026-05-20)

| Test | Validates | Result |
|------|-----------|--------|
| Registry: `cosmos_base` resolves to `DreamDojoTower` | Same class, alias only | **PASS** (AST verified) |
| `Pi0Config` with `vega3d_tower_name="cosmos_base"` auto-derives `feat_dim=2048` | No explicit setting needed | **PASS** (AST verified) |
| `pi05_libero_lora_cosmos_base` config exists and parses | In-process tower config | **PASS** (AST verified) |
| `pi05_libero_lora_cosmos_base_precomp` config exists and parses | Precomputed features config | **PASS** (AST verified) |
| `ensure_cosmos_base_checkpoint()` present in precompute script | HF download guidance | **PASS** (AST verified) |
| No `CosmosBaseTower` subclass exists | Dropped per architecture identity finding | **PASS** (grep confirmed) |
| Comments say "condition mask" not "action channel" | Corrected per NVIDIA source investigation | **PASS** (grep confirmed) |

### Follow-ups

- Runtime config parse verification (requires torch environment)
- Precompute smoke test with real checkpoint
- In-process training memory profiling on 48GB GPU
- Online probe with base Cosmos checkpoint (after HF download)

---

## Phase 6: DreamDojo as Third Generative-Tower Backbone

### Sub-Phase 6.4 — Spatial-Grid Adaptation (2026-05-19)

**Goal**: Add input resize to guarantee 256-token output regardless of input image size. Bilinear interpolate to `input_resolution=256` before VAE encoding.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1. Offline mode (no checkpoint).

### Completed Tests — Offline (8)

| Test | Validates | Result |
|------|-----------|--------|
| `encode(torch.zeros(1, 3, 224, 224))` → `[1, 256, 2048]` | Upscale 224→256, correct token count | **PASS** |
| `encode(torch.zeros(1, 3, 256, 256))` → `[1, 256, 2048]` | Native resolution, no-op resize | **PASS** |
| `encode(torch.zeros(1, 3, 480, 480))` → `[1, 256, 2048]` | Downscale 480→256, correct token count | **PASS** |
| `tower.check_output(...)` passes BaseTower ABC | output_shape=(2,256,2048), frozen=True | **PASS** |
| Offline encode returns `[B, 256, 2048]` | No regression to offline fallback | **PASS** |
| policy_utils injects `input_resolution=256` for dreamdojo | `setdefault` produces correct kwargs | **PASS** |
| `feat_dim` = 2048 | Unchanged from 6.2/6.3 | **PASS** |
| Batch dim: `encode(torch.zeros(4, 3, 300, 300))` → `[4, 256, 2048]` | Batch + non-square resize | **PASS** |

### Existing Test Suites

| Test | Validates | Result |
|------|-----------|--------|
| `scripts/test_tower.py --offline` | Syntax (19/19), ABC contract, import graph, registry, diagnostics | **PASS** |
| All 32 TrainConfigs parse | No regressions to existing configs | **PASS** |

### Code Changes

| Change | File | Lines |
|--------|------|-------|
| Added `import torch.nn.functional as F` | `dreamdojo_tower.py` | 15 |
| Added `F.interpolate` resize before VAE encode | `dreamdojo_tower.py` | 183-190 |
| Added `input_resolution=256` default for dreamdojo | `policy_utils.py` | 73-74 |

### Online Tests — PENDING (checkpoint not available)

| Test | Validates | Status |
|------|-----------|--------|
| Memory < 14GB at 256×256 | Forward fits alongside Pi0 | **PENDING** |
| Forward time < 400ms per batch-of-1 | Order-of-magnitude timing | **PENDING** |

---

### Sub-Phase 6.8 — Documentation + Cleanup (2026-05-19)

**Goal**: Final documentation pass — update CHANGELOG, TEST_STATUS, PHASE6_PLAN with results from 6.5–6.7. Mark all sub-phase status markers DONE.

### Completed Tests (3)

| Test | Validates | Result |
|------|-----------|--------|
| PHASE6_PLAN.md all markers → DONE | All 9 sub-phases (6.0–6.8) marked complete | **PASS** |
| CHANGELOG.md entries for 6.5, 6.6, 6.7, 6.8 | Each sub-phase has goal, files, key decisions, validation | **PASS** |
| TEST_STATUS.md entries for 6.5, 6.6, 6.7, 6.8 | Test tables present for all remaining sub-phases | **PASS** |

---

### Sub-Phase 6.7 — test_tower.py Validation (2026-05-19)

**Goal**: Run existing `scripts/test_tower.py --offline` to verify DreamDojo tower is picked up by AST-based registry validation. No code changes needed.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1. Offline mode.

### Completed Tests (5)

| Test | Validates | Result |
|------|-----------|--------|
| Syntax validation (19/19 files) | All tower package files parse cleanly | **PASS** |
| BaseTower ABC contract | DreamDojoTower satisfies encode/feat_dim/freeze/check_output | **PASS** |
| Import graph acyclic | No circular dependencies in tower package | **PASS** |
| TOWER_REGISTRY keys = `{"vae", "wan_t2v", "dreamdojo"}` | DreamDojo present in registry | **PASS** |
| Diagnostics pass | check_output returns correct shape/frozen status | **PASS** |

### Existing Test Suites

| Test | Validates | Result |
|------|-----------|--------|
| `scripts/test_tower.py --offline` | Full offline validation suite | **PASS** |
| All 34 TrainConfigs parse | No regressions after 6.5/6.6 additions | **PASS** |

---

### Sub-Phase 6.6 — Camera-Choice Config (2026-05-19)

**Goal**: Add `pi05_b1k_dreamdojo_wrist` TrainConfig entry targeting wrist cameras for egocentric DreamDojo fusion.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1.

### Completed Tests (4)

| Test | Validates | Result |
|------|-----------|--------|
| `get_config('pi05_b1k_dreamdojo_wrist')` parses | Config entry exists and resolves | **PASS** |
| `vega3d_cameras == ("left_wrist_0_rgb", "right_wrist_0_rgb")` | Wrist cameras correctly configured | **PASS** |
| `project_name == "B1K-DreamDojo-Wrist"` | Distinct W&B project name | **PASS** |
| All 34 TrainConfigs parse (32 original + dreamdojo + dreamdojo_wrist) | No regressions | **PASS** |

### Code Changes

| Change | File | Notes |
|--------|------|-------|
| Added `pi05_b1k_dreamdojo_wrist` TrainConfig block | `config.py` | Mirrors `pi05_b1k_dreamdojo` with wrist cameras |

---

### Sub-Phase 6.5 — TrainConfig Integration (2026-05-19)

**Goal**: Add `pi05_b1k_dreamdojo` TrainConfig entry mirroring `pi05_b1k_vega3d` but using DreamDojo as generative tower.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1.

### Completed Tests (4)

| Test | Validates | Result |
|------|-----------|--------|
| `get_config('pi05_b1k_dreamdojo')` parses | Config entry exists and resolves | **PASS** |
| `vega3d_tower_name == "dreamdojo"` | Correct tower backend selected | **PASS** |
| `vega3d_tower_kwargs` includes `checkpoint_dir`, `variant`, `input_resolution` | DreamDojo-specific kwargs present | **PASS** |
| All 33 TrainConfigs parse (32 original + dreamdojo) | No regressions after addition | **PASS** |

### Code Changes

| Change | File | Notes |
|--------|------|-------|
| Added `pi05_b1k_dreamdojo` TrainConfig block | `config.py` | Same data/optimizer/freeze as `pi05_b1k_vega3d`, tower swapped to dreamdojo |

---

### Sub-Phase 6.3 — Null-Text Forward Pass (2026-05-19)

**Goal**: Replace dummy `encode()` output with real forward pass through frozen Cosmos-Predict2.5-2B transformer. Zero text embeddings, zero action channel, flow-matching noise at timestep 300, hook at block 20 (70% depth).

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1. Offline mode (no DreamDojo checkpoint or Cosmos VAE on disk).

### Completed Tests — Offline (9)

| Test | Validates | Result |
|------|-----------|--------|
| Registry keys unchanged | `{"vae", "wan_t2v", "dreamdojo"}` still present | **PASS** |
| `DreamDojoTower("dummy")` instantiates (offline) | No-checkpoint fallback works, `transformer=None`, `vae=None` | **PASS** |
| `encode()` shape `[2, 256, 2048]` at 256×256 | Output contract: 256 tokens, feat_dim=2048 | **PASS** |
| `feat_dim` = 2048 | Config-based introspection: 16 heads × 128 dim | **PASS** |
| Trainable params = 0 | Tower frozen (offline: no model params) | **PASS** |
| `variant="student"` raises `NotImplementedError` | Student variant refused | **PASS** |
| `action_regime="averaged"` raises `NotImplementedError` | Non-null regimes gated | **PASS** |
| `feat_block_idx=28` raises `ValueError` | Out-of-range block index caught | **PASS** |
| `tower.online` = False | Offline mode correctly reported | **PASS** |

### Existing Test Suites

| Test | Validates | Result |
|------|-----------|--------|
| `scripts/test_tower.py --offline` | Syntax (19/19), ABC contract, import graph, registry, diagnostics | **PASS** |
| All 32 TrainConfigs parse | No regressions to existing configs | **PASS** |

### Code Change

| Change | File | Line |
|--------|------|------|
| Added shape assert on hook output: `assert feats.ndim == 3 and feats.shape[-1] == self._feat_dim` | `dreamdojo_tower.py` | 233-235 |

### Online Tests — PENDING (checkpoint not available)

| Test | Validates | Status |
|------|-----------|--------|
| `tower.encode(torch.randn(2, 3, 256, 256))` returns `[2, 256, 2048]` | Real forward produces correct shape | **PENDING** |
| `output.std() > 1e-4` | Non-degenerate features | **PENDING** |
| Output dtype = bf16, device = cuda | Dtype/device match constructor args | **PENDING** |
| Memory < 12GB on 48GB GPU | Forward fits alongside Pi0 | **PENDING** |
| Forward time < 200ms per batch-of-1 | Order-of-magnitude timing check | **PENDING** |

### Plan Correction

| Issue | Original Plan (6.3 tests) | Corrected Value |
|-------|---------------------------|-----------------|
| Expected shape at 224×224 | `[2, 49, feat_dim]` (assumed 32× stride) | `[2, 196, feat_dim]` (actual 16× stride: 8× VAE + 2× patchify) |

---

### Sub-Phase 6.2 — Real Loader + feat_dim Introspection (2026-05-12)

**Goal**: Replace scaffold with real DreamDojo model loading, introspect feat_dim from loaded config.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, diffusers 0.37.1, RTX 6000 Ada 49GB.

### Completed Tests (9)

| Test | Validates | Result |
|------|-----------|--------|
| Registry keys unchanged | `{"vae", "wan_t2v", "dreamdojo"}` still present | **PASS** |
| `DreamDojoTower("dummy")` instantiates (offline mode) | No-checkpoint fallback works, `transformer=None` | **PASS** |
| `encode()` shape `[1, 256, 2048]` | Output contract unchanged from 6.1 | **PASS** |
| `variant="student"` raises `NotImplementedError` | Student variant still refused | **PASS** |
| `feat_dim` = 2048 from config (offline) | Config-based introspection: 16 heads × 128 dim | **PASS** |
| `feat_dim` consistent across multiple constructions | Same value every time | **PASS** |
| `freeze()` makes all params non-trainable | 0 trainable params (offline: no model; online: all frozen) | **PASS** |
| `num_blocks` = 28 | Block count matches Cosmos-Predict2.5-2B architecture | **PASS** |
| `feat_block_idx=28` raises `ValueError` | Out-of-range block index caught | **PASS** |

### Online Loading Test (synthetic checkpoint)

| Test | Validates | Result |
|------|-----------|--------|
| Round-trip: diffusers model → reverse key map → NVIDIA format → save → load | 570 keys matched, 0 missing architecture keys | **PASS** |
| Action keys skipped | `action_embedder_*` keys reported as unexpected, not loaded | **PASS** |
| `freeze()` online: 570 params, 0 trainable | All loaded params frozen | **PASS** |
| `encode()` returns correct shape from online tower | `[2, 256, 2048]` — same contract as offline | **PASS** |

### Additional Validation

| Test | Validates | Result |
|------|-----------|--------|
| All 32 TrainConfigs parse | No regressions to existing configs | **PASS** |

### Architecture Facts Confirmed

| Property | Value | Source |
|----------|-------|--------|
| hidden_size | 2048 (16 × 128) | DreamDojo DCP metadata + diffusers instantiation |
| num_layers | 28 | DCP metadata: blocks 0–27 |
| in_channels | 17 (16 VAE + 1 action) | x_embedder weight shape (2048, 72) = 18×4 |
| out_channels | 16 | final_layer weight shape (64, 2048) = 16×4 |
| total params | 1.96B (2B config) | diffusers meta-device instantiation |

---

### Sub-Phase 6.1 — Skeleton DreamDojoTower Scaffold (2026-05-12)

**Goal**: Create placeholder `DreamDojoTower(BaseTower)`, register in TOWER_REGISTRY, verify end-to-end.

**Environment**: Python 3.10, PyTorch 2.7.1+cu126, CPU-only (no checkpoint needed).

### Completed Tests (7)

| Test | Validates | Result |
|------|-----------|--------|
| `TOWER_REGISTRY.keys()` shows `{"vae", "wan_t2v", "dreamdojo"}` | Registry contains exactly 3 keys after adding dreamdojo | **PASS** |
| `DreamDojoTower(checkpoint_dir="dummy")` instantiates | Scaffold construction works without a real checkpoint | **PASS** |
| `tower.encode(torch.zeros(1, 3, 224, 224))` returns shape `[1, 256, 2048]` | Output contract matches BaseTower (256 tokens, feat_dim=2048) | **PASS** |
| `tower.encode(torch.zeros(4, 3, 224, 224))` returns shape `[4, 256, 2048]` | Batch dimension handled correctly | **PASS** |
| `DreamDojoTower(variant="student")` raises `NotImplementedError` | Student variant explicitly refused with clear message | **PASS** |
| `tower.feat_dim == 2048` | Property returns expected placeholder value | **PASS** |
| `tower.check_output(images)` returns correct diagnostics | Inherited BaseTower method works: output_shape, feat_dim, frozen=True | **PASS** |

### Additional Validation

| Test | Validates | Result |
|------|-----------|--------|
| `scripts/test_tower.py --offline` | Syntax (19/19), ABC contract, import graph, registry, diagnostics — all pass | **PASS** |
| All 32 TrainConfigs parse | No regressions to existing configs | **PASS** |

---

### Sub-Phase 6.0 — Investigation & Locked Decisions (2026-05-12)

**Goal**: Investigate Cosmos-Predict2.5-2B architecture, identify blockers, lock design decisions.

**Environment**: Same as Phase 4 (Python 3.10, PyTorch 2.7.1+cu126). Web research against diffusers docs, HuggingFace model cards, NVIDIA repos.

### Completed Tests (3)

| Test | Validates | Result |
|------|-----------|--------|
| PHASE6_INVESTIGATION.md renders cleanly, all internal references resolve | Doc is complete and well-structured (287 lines) | **PASS** |
| Locked decisions cross-reference sources | Decision 1 cites HF diffusers docs, Decision 3 cites Cosmos-Tokenizer repo `CV8x8x8` naming | **PASS** |
| Dependency graph is acyclic | Sub-phase deps form a DAG: 6.0→6.1→6.2→6.3→6.4→{6.5,6.7}→6.6→6.8 | **PASS** |

### Critical Correction Found

| Issue | Original Plan | Corrected Value | Source |
|-------|--------------|-----------------|--------|
| input_resolution | 448 (yields 784 tokens) | **256** (yields 256 tokens) | Cosmos VAE 8x spatial + patch (1,2,2) = 16x stride |

---

### Phase 6 Planning — Plan committed (2026-05-12)

**Goal**: Commit `docs/PHASE6_PLAN.md` with full sub-phase breakdown. No code changes.

**Environment**: Same as Phase 4 (Python 3.10, PyTorch 2.7.1+cu126).

### Completed Tests (1)

| Test | Validates | Result |
|------|-----------|--------|
| Plan doc renders cleanly, structure matches PHASE4_PLAN.md conventions | Phase 6 plan is valid and ready for execution | **PASS** |

### Follow-ups (pending Sub-Phases 6.0-6.8)

| Item | Notes |
|------|--------|
| Investigation doc | Sub-phase 6.0 — write `PHASE6_INVESTIGATION.md` with blocker analysis and locked decisions. |
| Skeleton scaffold | Sub-phase 6.1 — `DreamDojoTower(BaseTower)` placeholder in TOWER_REGISTRY. |
| Real loader + feat_dim | Sub-phase 6.2 — load Cosmos-Predict2.5-2B, introspect hidden dim. **Requires checkpoint download.** |
| Null-action forward pass | Sub-phase 6.3 — real `encode()` with zero actions/text. **Highest risk sub-phase.** |
| Spatial-grid adaptation | Sub-phase 6.4 — 448x448 input -> 256 tokens output. |
| TrainConfig integration | Sub-phase 6.5 — `pi05_b1k_dreamdojo` config entry + smoke test. |
| Camera-choice config | Sub-phase 6.6 — `pi05_b1k_dreamdojo_wrist` for egocentric cameras. |
| test_tower.py validation | Sub-phase 6.7 — existing script validates new backbone. |
| Documentation + cleanup | Sub-phase 6.8 — final docs pass. |

---

## Phase 4: Adapter Training

### Sub-Phase 4.1 — Data config + TrainConfig (2026-05-04)

**Goal**: Port `LeRobotB1KDataConfig` and create `pi05_b1k_vega3d` TrainConfig entry.

**Environment**: Python 3.10, openpi-Vega3D venv, same as Phase 3.

### Completed Tests (4)

| Test | Validates | Result |
|------|-----------|--------|
| Config parse (`get_config('pi05_b1k_vega3d')`) | Config entry exists, all fields resolve, model_type=PI05, use_vega3d=True, cameras=("base_0_rgb",), force_gate=None | **PASS** |
| Data factory (`config.data.create(...)`) | `LeRobotB1KDataConfig` produces valid `DataConfig` with RepackTransform, B1kInputs/B1kOutputs, 4 model transforms, use_quantile_norm=True, 22 tasks, 190 episodes | **PASS** |
| No regressions (all 32 configs) | All existing configs parse, names unique, no import errors | **PASS** |
| Checkpoint path exists | `/workspace/RLinf/safetensors_ckpts/openpi_05_20251115_050323_9000_tor` on disk | **PASS** |

### Follow-ups (pending Sub-Phases 4.3–4.8)

| Item | Notes |
|------|--------|
| Adapter param count assertion | Sub-phase 4.3 — filter optimizer to only `P_gen`, `P_sem`, `fusion.*`; assert ~4M trainable. |
| Gate stats logging | Sub-phase 4.3 — log `g_mean`, `g_std`, histogram per step during training. |
| `to_unit_range` compile fix | Sub-phase 4.4 — remove `.item()` call; verify torch.compile runs clean. |
| Spatial dropout | Sub-phase 4.5 — `dropout_p=0.1` in training mode; unit test. |
| Smoke training run (500 steps) | Sub-phase 4.6 — loss decreases, gate spreads, no OOM. **Requires data.** |
| Convergence training | Sub-phase 4.7 — 10K-50K steps, val loss below baseline. **Requires data.** |
| Evaluation (300 rollouts) | Sub-phase 4.8 — 5 tasks × 20 rollouts × 3 methods. **Requires data + trained checkpoint.** |

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
