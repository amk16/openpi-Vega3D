# Phase 6: DreamDojo as Third Generative-Tower Backbone — Full Plan

**Last updated:** 2026-05-12
**Status:** In progress. Sub-phases 6.0–6.2 complete; 6.3 is next.

---

## What Phase 6 Is

Phase 6 registers DreamDojo as a third backbone in TOWER_REGISTRY alongside "vae" and "wan_t2v". The deliverable is a working `DreamDojoTower(BaseTower)` implementation that loads NVIDIA's Cosmos-Predict2.5-2B-teacher (DreamDojo's foundation backbone), produces feature tokens compatible with PaliGemma's 256-token spatial grid, and passes the same battery of tower-validation tests the VAE and WAN backbones pass.

**This is infrastructure, not training.** Phase 6 proves the backbone works structurally and passes inference. Adapter training (convergence, eval) is a future Phase 7 — mirroring how Phases 0-3 built VAE/WAN infrastructure and Phase 4 trains the VAE adapter.

### Phase Numbering Rationale

| Phase | Scope | Status |
|-------|-------|--------|
| 0-3 | Infrastructure (registry, env, policy, fusion) | Done |
| 4 | VAE adapter training | In progress (4.1 done) |
| 5 | WAN adapter training | Reserved per PHASE4_PLAN.md |
| **6** | **DreamDojo as third backbone** | **This plan** |
| 7 (future) | DreamDojo adapter training | Outside this plan |

---

## Locked Decisions

These will be established and justified in `docs/PHASE6_INVESTIGATION.md` (sub-phase 6.0). Do not revisit unless you hit a specific wall.

| # | Decision | Value | Why |
|---|----------|-------|-----|
| 1 | Checkpoint source | DreamDojo 2B pretrain (`nvidia/DreamDojo`) | DreamDojo's 44k hours of egocentric video gives robotics-relevant features. Same Cosmos-Predict2.5 architecture; load weights with `strict=False` to skip action-conditioning keys. Teacher only — student refused at construction. |
| 2 | Action regime | Null (Regime A) | Feed a zero 32-d x 4-stacked action tensor to the AdaLN slot. Passive geometric-prior extraction isolates the data-distribution effect from the action-conditioning effect. |
| 3 | Spatial-grid strategy | Input resolution 256 (corrected from plan's original 448) | Feed 256x256 so Cosmos's 8x VAE + 2x2 patchify = 16x total stride yields exactly 16x16 = 256 tokens. No pooling needed. Keeps `_fuse_camera` untouched. See PHASE6_INVESTIGATION.md Blocker 4 for correction rationale. |
| 4 | Layer-depth default | `feat_block_idx = round(0.7 * num_blocks)` | Matches VEGA-3D paper's ~70% fractional-depth claim. Note: existing WAN tower defaults to -1 (last block) — discrepancy worth flagging and fixing separately. |
| 5 | Camera selection | Support both base and wrist configs | `pi05_b1k_dreamdojo` uses base camera (apples-to-apples vs WAN baseline). `pi05_b1k_dreamdojo_wrist` uses wrist cameras (egocentric-prior-friendly viewpoint match). |

---

## Sub-Phase Status

```
6.0  Investigation               DONE     docs/PHASE6_INVESTIGATION.md
6.1  Skeleton scaffold            DONE     src/openpi_vega3d/towers/dreamdojo_tower.py
6.2  Real loader + feat_dim       DONE     dreamdojo_tower.py (replaces scaffold)
6.3  Null-action forward pass     TODO     dreamdojo_tower.py (real encode())
6.4  Spatial-grid adaptation      TODO     dreamdojo_tower.py + policy_utils.py
6.5  TrainConfig integration      TODO     src/openpi/training/config.py
6.6  Camera-choice config         TODO     src/openpi/training/config.py
6.7  test_tower.py validation     TODO     scripts/test_tower.py (likely no-op)
6.8  Documentation + cleanup      TODO     docs/PHASE6_PLAN.md, CHANGELOG, TEST_STATUS
```

### Dependency Graph

```
6.0  (investigation)
 │
 ▼
6.1  (skeleton scaffold)
 │
 ▼
6.2  (real loader + feat_dim introspection)
 │
 ▼
6.3  (null-action forward pass)
 │
 ▼
6.4  (spatial-grid adaptation)
 │
 ├────────────────────────┐
 ▼                        ▼
6.5  (TrainConfig)       6.7  (test_tower.py validation)
 │
 ▼
6.6  (camera-choice config)
 │
 └────────────────────────┐
                          ▼
                         6.8  (documentation + cleanup)
```

Sub-phases 6.5/6.6 and 6.7 can be done in parallel after 6.4 completes. All must be done before 6.8.

---

## Detailed Instructions Per Sub-Phase

### 6.0 — Investigation & Locked Decisions

**Goal:** Write `docs/PHASE6_INVESTIGATION.md` documenting the architectural delta, locked decisions, and dependency graph. Mirrors Phase 4.0's role. No code changes.

**Deliverable:** `docs/PHASE6_INVESTIGATION.md` (~300 lines) covering:
- Architectural delta vs WAN tower (refers to research-wiki for upstream theory)
- Five concrete blockers with `file:line` references
- Three locked decisions (checkpoint variant, action regime, spatial-grid strategy) with reasoning
- Dependency graph for sub-phases 6.1-6.8
- Open questions/risks list

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| Doc renders cleanly | Markdown parses; all internal links resolve | PASS |
| Locked decisions cross-reference wiki | Each decision cites the corresponding wiki page slug | PASS |
| Dependency graph is acyclic | Sub-phase deps form a DAG | PASS |

**Files touched:** `docs/PHASE6_INVESTIGATION.md` (new). No code.

---

### 6.1 — Skeleton DreamDojoTower Scaffold

**Goal:** Create a placeholder `DreamDojoTower(BaseTower)` class returning a dummy fixed-shape tensor (no real model load yet) and register it in TOWER_REGISTRY. Verify registration plumbing end-to-end without touching real NVIDIA weights.

**Why scaffold first:** Lets the rest of the integration (config plumbing, scripts) move forward without requiring a multi-GB checkpoint download. Phase 4 sub-phase 4.1 used the same scaffold-first pattern.

**Files to modify:**
- `src/openpi_vega3d/towers/dreamdojo_tower.py` (new) — `DreamDojoTower(BaseTower)` skeleton:
  - `__init__(self, checkpoint_dir, variant="teacher", action_regime="null", output_spatial=16, dtype="bf16")`
  - `feat_dim` property: returns hardcoded placeholder 2048 (TBD until 6.2 introspection)
  - `encode(self, images)` returns `torch.zeros(B, output_spatial**2, 2048)` with correct dtype/device
  - `freeze()` no-op (no real params to freeze yet)
  - `check_output(images)` inherits from BaseTower
  - Raise `NotImplementedError("variant='student' is not supported in Phase 6; use 'teacher'")` if `variant=="student"`
- `src/openpi_vega3d/towers/__init__.py` — add `"dreamdojo": DreamDojoTower` to TOWER_REGISTRY (lazy import)

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `from openpi_vega3d.towers import TOWER_REGISTRY` shows three keys | Registry contains exactly {"vae", "wan_t2v", "dreamdojo"} | PASS |
| `DreamDojoTower(checkpoint_dir="dummy")` instantiates | Scaffold construction works without a real checkpoint | PASS |
| `tower.encode(torch.zeros(1, 3, 224, 224))` returns shape `[1, 256, 2048]` | Output contract matches BaseTower | PASS |
| `DreamDojoTower(variant="student")` raises `NotImplementedError` | Student variant explicitly refused with clear message | PASS |
| `scripts/test_tower.py --offline` passes | Offline validation picks up new backbone in registry (via AST parse of `__init__.py`) | PASS |
| All existing 32 TrainConfigs parse | No regressions to other backbones | PASS |

---

### 6.2 — Real Loader + feat_dim Introspection (Blockers 1 + 4)

**Goal:** Replace scaffold with real DreamDojo/Cosmos-Predict2.5 loading. Introspect `feat_dim` (DiT hidden dim) from the loaded model. Still using dummy feature outputs from `encode()` — just confirm the model loads and we can read its config.

**Prerequisite (user action):** Download DreamDojo 2B pretrain from `nvidia/DreamDojo` (`2B_pretrain/iter_000140000/model/`). Convert DCP → `.pt` via cosmos-predict2.5's `convert_distcp_to_pt.py`. Place result in `ckpts/DreamDojo-2B/`.

**Files to modify:**
- `src/openpi_vega3d/towers/dreamdojo_tower.py`:
  - Add real loader: instantiate `CosmosTransformer3DModel` via diffusers for architecture, then load DreamDojo `.pt` weights with `strict=False` (extra action-conditioning keys skipped)
  - Introspect: `self._feat_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim`
  - Verify DiT block list at `transformer.transformer_blocks` (confirmed in investigation)
  - Load VAE separately for latent encoding
  - `freeze()` actually sets `requires_grad_(False)` on every parameter
  - `encode()` still returns dummy zeros for now (real forward in 6.3)
  - Log any missing/unexpected keys from `strict=False` load for debugging

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| Tower loads from real checkpoint dir | `DiffusionPipeline.from_pretrained(...)` succeeds; reports total param count | PASS |
| `feat_dim` populated from `pipe.transformer.config.hidden_size` | Resolves to actual hidden dim (typical for 2B is 1.5K-2.5K range); recorded in changelog | PASS |
| `feat_dim` is consistent across multiple construction calls | Same checkpoint -> same feat_dim every time | PASS |
| `freeze()` makes every parameter non-trainable | `sum(p.requires_grad for p in transformer.parameters()) == 0` | PASS |
| DiT block list is found and non-empty | `len(blocks) > 0` for whichever attribute path resolves | PASS |
| All scaffolded tests from 6.1 still pass | No regression | PASS |

**Risk:** Cosmos-Predict2.5 may not load cleanly via `DiffusionPipeline.from_pretrained` (less standard than WAN). Fallback: load via `cosmos-predict2.5` GitHub package's own loader. Document either way.

---

### 6.3 — Null-Text Forward Pass (Blockers 2 + 5)

**Goal:** Replace dummy `encode()` output with a real forward pass through the loaded Cosmos-Predict2.5 model, with zero-valued text embeddings. No action conditioning exists in the base model (see PHASE6_INVESTIGATION.md Decision 2). The forward runs end-to-end and produces real features — but the spatial grid is still native (e.g., 14x14 = 196 at 224x224 input); output_spatial adapter comes in 6.4.

**Files to modify:**
- `src/openpi_vega3d/towers/dreamdojo_tower.py`:
  - Real `encode()` implementation:
    1. VAE encode input images to latents
    2. Build zero action tensor of shape `[B, 4, 32]` (4 consecutive 32-d continuous latent actions per the DreamDojo paper)
    3. Build zero text embeddings of the shape Cosmos-Reason1 would produce (introspect or set to known-good shape)
    4. Build flow-matching noise: `tau = torch.tensor([300], dtype=long)` and `noisy_latents = scheduler.add_noise(latents, noise, tau)` (mirrors `wan_t2v_encoder.py` lines 186-204)
    5. Register forward hook on intermediate DiT block (default: `feat_block_idx = round(0.7 * num_blocks)` — per VEGA-3D paper's ~70% fractional depth)
    6. Run forward pass with all conditioning zeroed
    7. Reshape hook output to `[B, N_native_tokens, feat_dim]`
  - Add `action_regime` param check; raise `NotImplementedError` if not `"null"` in v1

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `tower.encode(torch.randn(2, 3, 224, 224))` returns shape `[2, 49, feat_dim]` | Forward pass produces non-trivial features at native grid | PASS |
| Output is non-zero / non-constant | `output.std() > 1e-4` (not degenerate) | PASS |
| Output dtype = bf16 / device = cuda | Matches dtype constructor arg | PASS |
| Tower is fully frozen during forward | `sum(p.grad is not None ...) = 0` after `loss.backward()` from downstream | PASS |
| Memory budget: forward pass uses < 12GB on a single 48GB GPU | Fits alongside Pi0 (~3.5B frozen) for end-to-end | PASS |
| Forward time: < 200ms per batch-of-1 at 224x224 | Order-of-magnitude check; not strict | PASS |
| `action_regime="averaged"` raises `NotImplementedError` | Future regimes explicitly gated | PASS |

**Risk:** The AdaLN action-input pathway in Cosmos-Predict2.5's diffusers wrapper may not expose a clean "action embedding" argument. The DreamDojo paper says actions are concatenated then summed into AdaLN, but the public Cosmos-Predict2.5 weights may not have action conditioning at all (it's a base model; action was added in DreamDojo's specialization). **Crucial verification at 6.2** — what does `pipe.transformer.forward` signature accept? If actions are not exposed, we either use DreamDojo's specialized checkpoint instead of base Cosmos, or zero-pad something that gets multiplied into AdaLN regardless.

---

### 6.4 — Spatial-Grid Adaptation (Blocker 3, option b)

**Goal:** Resolve the native-grid vs 256 token mismatch by running the tower at 2x input resolution. The fusion module gets exactly `[B, 256, feat_dim]`, identical in shape to what VAE and WAN towers produce, so `_fuse_camera` is untouched.

**Files to modify:**
- `src/openpi_vega3d/towers/dreamdojo_tower.py`:
  - Add `input_resolution: int = 256` constructor param
  - In `encode()`: resize input images to `(input_resolution, input_resolution)` via bilinear interpolation if they arrive at a different size
  - At 256x256 with Cosmos's 8x VAE + 2x2 patchify = 16x stride: 256/16 = 16x16 = 256 tokens exactly. No pooling needed.
  - If VAE compression turns out different (verified in 6.2), recalculate: `input_resolution = 16 × VAE_stride × patch_spatial_size`
- `src/openpi_vega3d/policy_utils.py` — add DreamDojo defaults for the loader, including `input_resolution=256`.

**Note on spatial math (corrected):** Investigation (PHASE6_INVESTIGATION.md Blocker 4) confirmed Cosmos uses 8x spatial VAE compression + (1,2,2) patch_size. Total stride = 16. Input 256x256 → 16x16 = 256 tokens with no pooling.

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `tower.encode(torch.randn(1, 3, 224, 224))` returns shape `[1, 256, feat_dim]` | Output matches PaliGemma's token count when caller feeds 224x224 (tower resizes internally to 256) | PASS |
| `tower.encode(torch.randn(1, 3, 256, 256))` returns shape `[1, 256, feat_dim]` | Caller can also feed 256 directly (native resolution) | PASS |
| `tower.check_output(...)` passes existing BaseTower ABC contract | Same contract VAE and WAN satisfy | PASS |
| Memory budget: forward at 256x256 uses < 14GB | Comparable to WAN at similar resolution | PASS |
| Forward time: < 400ms per batch-of-1 at 256x256 | Order-of-magnitude check | PASS |

---

### 6.5 — Integration into TrainConfig + Smoke Test

**Goal:** Create a new `pi05_b1k_dreamdojo` TrainConfig entry mirroring `pi05_b1k_vega3d` but with `vega3d_tower_name="dreamdojo"`. Verify full Pi0 model constructs end-to-end with DreamDojo plugged in, and one forward+backward dummy step doesn't error.

**Files to modify:**
- `src/openpi/training/config.py` — add `pi05_b1k_dreamdojo` TrainConfig entry. Copy the `pi05_b1k_vega3d` block; change `vega3d_tower_name="dreamdojo"`, `vega3d_tower_kwargs={"checkpoint_dir": ..., "variant": "teacher", "input_resolution": 256}`. Keep `vega3d_cameras=("base_0_rgb",)` for the smoke test (camera-choice analysis is sub-phase 6.6).

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `get_config('pi05_b1k_dreamdojo')` parses | Config entry exists, all fields resolve, `vega3d_tower_name="dreamdojo"` | PASS |
| `PI0Pytorch(config)` constructs with DreamDojo as tower | `model.spatial_tower` is a `DreamDojoTower` instance; `model.P_gen` is `nn.Linear(feat_dim, 2048)` | PASS |
| One forward step on dummy obs batch passes | `model(batch)` returns loss tensor, no shape errors | PASS |
| One backward step on dummy batch passes | `loss.backward()` no errors; gradients flow to `P_gen`, `P_sem`, `fusion.*`; do NOT flow to DreamDojo tower | PASS |
| Adapter param count is comparable to VAE adapter | ~4M-10M trainable order of magnitude (`P_gen` input dim may be larger than VAE's 4) | PASS |
| No regressions: all existing configs still parse | No accidental breakage of vae or wan_t2v configs | PASS |

---

### 6.6 — Camera-Choice Config (Wrist Cameras for Egocentric Priors)

**Goal:** Add an alternate config `pi05_b1k_dreamdojo_wrist` that applies the DreamDojo tower to the wrist cameras (`left_wrist_0_rgb`, `right_wrist_0_rgb`) rather than the base camera. DreamDojo's egocentric training distribution is a better viewpoint match for wrist cams than base cam.

**Files to modify:**
- `src/openpi/training/config.py` — add `pi05_b1k_dreamdojo_wrist`. Same as `pi05_b1k_dreamdojo` but `vega3d_cameras=("left_wrist_0_rgb", "right_wrist_0_rgb")`.

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `get_config('pi05_b1k_dreamdojo_wrist')` parses | New config entry resolves | PASS |
| `PI0Pytorch(config)` constructs with wrist cameras as VEGA-3D targets | `model._spatial_cameras = {"left_wrist_0_rgb", "right_wrist_0_rgb"}` | PASS |
| One forward+backward step on dummy data with wrist images | Wrist images flow to DreamDojo tower; base image flows to PaliGemma alone (un-fused) | PASS |
| Memory: forward+backward fits in 48GB | Two towers (one per wrist) at 256x256 + Pi0 backbone; should fit at batch 2 | PASS or escalate |

**Risk:** Running the tower twice (once per wrist cam) doubles tower-forward cost. At 256x256 (lighter than the plan's original 448), this is more feasible. May still need batch-size reduction or shared-tower-call optimization. Note the trade-off in changelog; don't block on it for v1.

---

### 6.7 — scripts/test_tower.py Validation

**Goal:** Run the existing tower-validation script against the new backbone and confirm it passes the same checks VAE and WAN already pass. This is an existing test entry point we want to plug into, not a new one.

**Files to modify:**
- `scripts/test_tower.py` — likely no changes needed. `--tower dreamdojo --checkpoint <path>` should work since the registry now contains the key. If the offline validation's AST-based registry check at lines 107-114 doesn't pick up the new key automatically, update the expected-keys set.

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| `python scripts/test_tower.py --offline` | Offline validation: registry key, BaseTower ABC compliance | PASS |
| `python scripts/test_tower.py --tower dreamdojo --checkpoint <path>` | Online validation: loads real checkpoint, runs encode, validates output shape `[B, 256, feat_dim]` and frozen state | PASS |
| Same script with `--tower wan_t2v` and `--tower vae` | Regression — existing backbones still pass | PASS |

---

### 6.8 — Documentation + Cleanup

**Goal:** Update this plan with final status, update `docs/CHANGELOG.md` with cumulative Phase 6 summary, mark all Phase 6 sub-phases as done in `docs/TEST_STATUS.md`.

**Files to modify:**
- `docs/PHASE6_PLAN.md` — mark all sub-phases done
- `docs/CHANGELOG.md` — top-level Phase 6 summary block
- `docs/TEST_STATUS.md` — all Phase 6 sub-phases marked done

**Tests:**

| Test | Validates | Expected |
|------|-----------|----------|
| All Phase 6 docs render cleanly | Markdown parses; internal links resolve | PASS |
| Changelog has Phase 6 block matching existing style | Sections: Files Modified, Key Decisions, Validation | PASS |
| Test status table shows full Phase 6 history | Each sub-phase has its test table | PASS |

---

## Live-Doc Maintenance (per sub-phase, every sub-phase)

After every sub-phase merge:

1. **`docs/CHANGELOG.md`** — append a new sub-phase block under "Phase 6" using the existing template:
   - Sub-Phase 6.X — Goal (YYYY-MM-DD)
   - "Goal" paragraph
   - "Files Modified" table
   - "Key Decisions and Reasoning" enumeration
   - "Validation" block

2. **`docs/TEST_STATUS.md`** — under "## Phase 6" (newest at top), add the sub-phase's "Completed Tests" table + "Follow-ups" list. Use the same Test | Validates | Result columns.

3. **`docs/PHASE6_PLAN.md`** — update the Sub-Phase Status block showing markers.

---

## Files to Be Modified or Created (Full List)

### New files
| File | Purpose | Approx. size |
|------|---------|-------------|
| `src/openpi_vega3d/towers/dreamdojo_tower.py` | DreamDojo backbone implementation | ~150 lines |
| `docs/PHASE6_INVESTIGATION.md` | Locked decisions + blocker analysis | ~300 lines |

### Modified files
| File | Change | Scope |
|------|--------|-------|
| `src/openpi_vega3d/towers/__init__.py` | Register "dreamdojo" in TOWER_REGISTRY | 1 line (lazy import) |
| `src/openpi/training/config.py` | Two new TrainConfig entries: `pi05_b1k_dreamdojo`, `pi05_b1k_dreamdojo_wrist` | ~180 lines |
| `src/openpi_vega3d/policy_utils.py` | Add DreamDojo defaults | ~20 lines |
| `scripts/test_tower.py` | Likely no-op; verify registry key picks up automatically | Verify only |
| `docs/CHANGELOG.md` | Eight sub-phase entries + Phase 6 summary block | ~400 lines over phase |
| `docs/TEST_STATUS.md` | Eight sub-phase test tables | ~200 lines over phase |

### Outside scope (no edits expected)
| File | Why |
|------|-----|
| `src/openpi/models_pytorch/pi0_pytorch.py` | Fusion is tower-agnostic |
| `src/openpi/models_pytorch/adaptive_gated_fusion.py` | No changes needed |
| `src/openpi_vega3d/towers/wan_t2v_encoder.py` | Untouched |

---

## Verification (Phase 6 Acceptance Criteria)

Phase 6 is complete when all of the following hold:

1. `from openpi_vega3d.towers import TOWER_REGISTRY` shows `{"vae", "wan_t2v", "dreamdojo"}` (three keys)
2. `scripts/test_tower.py --tower dreamdojo --checkpoint <path>` exits with PASS
3. `get_config('pi05_b1k_dreamdojo')` and `get_config('pi05_b1k_dreamdojo_wrist')` both parse
4. `PI0Pytorch` with `vega3d_tower_name="dreamdojo"` runs one forward+backward step on a dummy batch without errors
5. Adapter parameter count is within the same order of magnitude as the VAE adapter (~4M-10M trainable)
6. All existing vae and wan_t2v configs still parse and load (no regressions)
7. `docs/PHASE6_INVESTIGATION.md`, `docs/PHASE6_PLAN.md`, `docs/CHANGELOG.md`, `docs/TEST_STATUS.md` all updated with Phase 6 content matching repo conventions
8. The TEST_STATUS Phase 6 section shows every sub-phase 6.0-6.8 as done with passing test tables

---

## Known Risks and Mitigations

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|-----------|
| 1 | **Cosmos-Predict2.5 base has no AdaLN action conditioning.** Action conditioning may only exist in DreamDojo's specialization. | High | Sub-phase 6.2 must verify which forward-signature the loaded model accepts. If base Cosmos doesn't take an action input, use DreamDojo's specialized checkpoint instead. |
| 2 | **`pipe.transformer.config.hidden_size` may not be the right attribute.** Diffusers wrappers vary. | Medium | Verify in 6.2 by `print(pipe.transformer)` and `print(pipe.transformer.config)`. |
| 3 | **DiT block list attribute path is uncertain.** `transformer_blocks` vs `blocks` vs `layers`. | Medium | Verify empirically in 6.2. |
| 4 | **Spatial compression ratio is unverified.** Plan assumes 4x16x16 (like WAN), but Cosmos may differ. | Medium | Sub-phase 6.2 must introspect the VAE config. Recalculate `input_resolution` if compression ratio differs. |
| 5 | **Memory budget for wrist-camera dual-tower forward.** Two tower instances at 448x448 + Pi0. | Medium | Size in 6.6. If tight, reduce batch size or share tower instance across cameras. |
| 6 | **Wrist-camera rollouts may have non-egocentric content** in some episodes. | Low | Camera choice helps on average; doesn't guarantee per-episode improvement. Eval is Phase 7's problem. |

---

## Effort Estimate

| Sub-phase | Active engineering time | Wall clock | Notes |
|-----------|----------------------|-----------|-------|
| 6.0 | 2-3 hours | 2-3 hours | Investigation + doc writing |
| 6.1 | 1-2 hours | 1-2 hours | Well-understood pattern from Phase 1 |
| 6.2 | 2-4 hours | 2-4 hours + download | Checkpoint download + loader debugging |
| 6.3 | 3-6 hours | 3-6 hours | Biggest risk area: action-conditioning uncertainty |
| 6.4 | 1-2 hours | 1-2 hours | Straightforward spatial math |
| 6.5 | 1-2 hours | 1-2 hours | Config copy + smoke test |
| 6.6 | <1 hour | <1 hour | Config copy with camera swap |
| 6.7 | <1 hour | <1 hour | Run existing script |
| 6.8 | 1-2 hours | 1-2 hours | Doc writing |
| **Total** | **~12-22 hours** | **~12-22 hours + download** | Sub-phase 6.3 is the time-risk driver |

---

## Where to Find More Detail

| Document | What it covers |
|----------|---------------|
| `docs/PHASE6_INVESTIGATION.md` | Full investigation: blockers, locked decisions, dependency graph (after 6.0) |
| `docs/PHASES_0_TO_3.md` | How the tower infrastructure was built (Phases 0-3 retrospective) |
| `docs/PHASE4_PLAN.md` | VAE adapter training plan (parallel effort) |
| `docs/CHANGELOG.md` | Per-file change log with reasoning |
| `docs/TEST_STATUS.md` | Every test result across all phases |
