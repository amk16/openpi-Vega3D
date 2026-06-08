# Phase 8: WAN Fusion Fidelity Fixes — Full Plan

**Last updated:** 2026-06-07
**Status:** IMPLEMENTED (all sub-phases 8.0–8.6, branch `fidelity-fixes`, 2026-06-07). Local verification done (syntax/lint/math simulations); pytest suite + remote diagnostics pending first CI run / remote session — see `docs/TEST_STATUS.md` Phase 8.
**Provenance:** 2026-06-04/05 fidelity audit (research-wiki: `vega3d-fidelity-audit`, `wan-fusion-fidelity-breaks`). All claims verified at HEAD `747a5c1`; VEGA reference claims byte-verified against raw files fetched from github.com/H-EmbodVis/VEGA-3D. **2026-06-07 pre-implementation review pass:** every code claim in this plan independently re-verified at HEAD; one gap found and folded in (S3 variant-tag collision, see 8.1), test specs hardened (CI-safe geometry test, new JAX↔torch parity test), two out-of-scope findings recorded as known-issues in 8.6 (user declined scope expansion).

---

## What Phase 8 Is

Phase 8 fixes the two confirmed fidelity breaks in the WAN→PaliGemma gated fusion: (1) token-grid misregistration from letterbox padding, and (2) raw-stream blend math plus an under-powered generative projector. Both breaks are present in the code that produced every published WAN number (FFT+WAN 35.3% vs FFT 42% on the swap suites). Until they are fixed and the run repeated once, "richer features genuinely don't help" cannot be distinguished from "the features arrived misaligned and off-scale."

**This is fixes + diagnostics + config, not regen/training.** Phase 8 delivers code changes, diagnostic scripts, and a regen-ready training config. Cache regeneration, the fidelity-fixed training run, and the discriminating eval are Phase 9, outside scope. Break 3 from the audit (train/eval noise mismatch + duplicated eval code path) is **explicitly deferred** — the Phase 9 regen keeps the existing seeded-noise behavior. Accepted risk: if Break 3's noise fix is needed later, a second regen is required.

### Phase Numbering Rationale

| Phase | Scope | Status |
|-------|-------|--------|
| 0-3 | Infrastructure (registry, env, policy, fusion) | Done |
| 4 | VAE adapter training | Done |
| 5 | WAN adapter training | Done on main (LIBERO) |
| 6 | DreamDojo as third backbone | Done |
| 7 | DreamDojo training setup | Done |
| **8** | **WAN fusion fidelity fixes** | **This plan** |
| 9 (future) | Cache regen + fidelity-fixed run + discriminating eval | Outside this plan |

---

## Locked Decisions

| # | Decision | Value | Why |
|---|----------|-------|-----|
| 1 | Break-1 fix | Content-region pooling (NOT cover-crop, NOT square canvas) | Pillarbox = exactly 11 tokens/side → content region is an exact 30×30 square (clean token boundaries). Cover-crop destroys 42.3% of image rows for square→16:9 (verified arithmetic) and moves the misregistration to rows. Square canvases are OOD for the 480P-only WAN 1.3B (official model card) |
| 2 | Break-2 fix | Normed blend + mlp2x_gelu P_gen | Matches VEGA's deployed `feature_fusion.py:217-221` (byte-verified) — the code behind their published numbers; field-standard (SD-DINO arXiv 2305.15347 normalizes both streams "to align their scales and distributions"; no surveyed work blends raw heterogeneous streams) |
| 3 | Flag defaults | Legacy behavior preserved; fixes opt-in via new flags | Existing configs must keep producing features identical to their caches (reproducibility of published runs); the new fidelity-fix config opts in explicitly |
| 4 | P_sem | Stays absent (`vega3d_use_p_sem=False`) | Minimal-delta principle: PaliGemma's own multimodal projector already maps SigLIP tokens into the LLM space; reinstating P_sem is a Phase-9+ ablation, not a fidelity requirement |
| 5 | P_gen MLP init | Default nnx init (no zero-init) | Gate warmup starts at g=1.0 (pure SigLIP), already protecting early training; matches how the current Linear P_gen initializes |
| 6 | Raw blend retention | Kept behind the flag (off = legacy) | The raw blend is VEGA's paper Eq. 8 verbatim — the paper and the deployed code disagree with each other, and our port matched the paper. Keeping it documents how the break happened and gives a paper-faithful ablation cell |
| 7 | Branch | `fidelity-fixes` off `747a5c1` | Teammates have in-flight runs referencing main/HEAD |

---

## Sub-Phase Status

```
8.0  Diagnostic scripts            DONE bf17c1a   scripts/diagnose_wan_fidelity.py (new)
8.1  Break-1 fix: content pooling  DONE 94e7cab   wan_t2v_encoder.py, common.py, wan_tower.py, precompute_tower_features.py, sync_tower_features_to_s3.sh
8.2  Break-2a fix: normed blend    DONE ed47cb7   models/adaptive_gated_fusion.py, models_pytorch/adaptive_gated_fusion.py
8.3  Break-2b fix: P_gen MLP       DONE bb9c059   pi0.py, pi0_pytorch.py, pi0_config.py
8.4  Fidelity-fix training config  DONE d5baf00   src/openpi/training/config.py
8.5  Tests                          DONE 105b52e   scripts/test_tower.py, scripts/diagnose_wan_fidelity_test.py, src/openpi/models/pi0_test.py
8.6  Documentation                  DONE (this commit)  docs/CHANGELOG.md, docs/TEST_STATUS.md, output_spatial 14→16 (WAN files)
```

### Dependency Graph

```
8.0  Diagnostics (independent — runs on remote against existing cache/ckpt)

8.1        8.2        8.3        (independent of each other)
content    normed     P_gen
pooling    blend      MLP
 └──────────┴──────────┘
            │
            ▼
           8.4  Fidelity-fix config (consumes all three flags)
            │
            ▼
           8.5  Tests
            │
            ▼
           8.6  Documentation
```

---

## Detailed Changes Per Sub-Phase

### 8.0 — Diagnostic scripts (pre-fix evidence)

**Reason:** Confirm both break mechanisms empirically in the trained artifacts before changing code — and produce the before/after comparison Phase 9 needs.
**Source:** Diagnostics specced in the audit (research-wiki `wan-fusion-fidelity-breaks`); existing script patterns `scripts/probe_wan.py`, `scripts/inspect_gate_weights.py`.
**Problem:** Both breaks are code-confirmed but not yet measured in the trained cache/checkpoint (cache and checkpoints live on the remote).
**What the fix does:** Adds a standalone diagnostic script with two subcommands, runnable on the remote.

New `scripts/diagnose_wan_fidelity.py`:
1. `column-energy` — loads N cached `ep_*.safetensors`, reshapes to 16×16×1536, prints/plots per-column mean feature energy. Prediction if Break 1 is live in the trained cache: conspicuously distinct energy in the ~3 outermost columns each side (pure black-bar tokens; pool-bin math says exactly 3 pure-pad + 1 mixed per side). Doubles as end-to-end cache-provenance validation.
2. `norm-ratio` — at a checkpoint (or at init), logs ‖f_gen‖₂ / ‖f_sem‖₂ per token batch. A ratio far from 1 (×3 or more either way) confirms the Break-2 scale-mismatch mechanism is live.

Deliberately out of scope: a `noise-variance` subcommand (the Break-3 5-seed sizing diagnostic) would be ~30 lines on this same scaffolding, but Break 3 is outside Phase 8's locked scope — trivial to add later if Phase 9's result motivates it.

### 8.1 — Break-1 fix: content-region pooling

**Reason:** WAN features must describe the same image regions as the SigLIP tokens they are fused with, token-for-token.
**Source:** Audit Break 1. Our code: `wan_t2v_encoder.py:116-124` (letterbox via `common.py:69-97`, `scale=min`, pad −1.0), `:330-350` and `:215-229` (pooling of the padded grid in both forward paths); default `832*480` (`wan_tower.py:32`) confirmed live in the headline config (`config.py:1155-1204` sets no `size` override). Reference: VEGA `common.py:60` cover-crop with `scale=max`, **no pad path in the file** (byte-verified); the no-pad-tokens discipline is universal in the literature (DIFT 2306.03881, SD-DINO 2305.15347, Diffusion Hyperfeatures 2305.14334, VPP 2412.14803 — all square/cover inputs to a common grid).
**Problem:** A square 224² frame letterboxed into 832×480 leaves 42.3% black pillarbox. The 30×52 token grid is avg-pooled to 16×16 **including the bars**: the outer ~3 pooled columns per side are pad, and content occupies ~9.2 of 16 columns — while SigLIP's 16 columns span the full frame. The token-wise fusion then combines different image locations, violating the correspondence contract its own docstring states (`adaptive_gated_fusion.py:24-27`). (Note: cover-cropping to literally match VEGA is NOT the fix here — for square sources into a 16:9 canvas it destroys 42.3% of rows and moves the misregistration to the row axis. VEGA's preprocessing is only valid when source aspect ≈ canvas aspect.)
**What the fix does:** Pools the 16×16 output from **only the content region** of the token grid. The pillarbox is exactly 11 tokens per side (176 px / 16 px-per-token), so the content region is an exact 30×30 square (token columns 11–40) — full content preserved, both axes registered against SigLIP, and WAN stays at its trained 832×480 canvas (no OOD risk).

Changes:
1. `src/openpi_vega3d/towers/common.py`: new helper `letterbox_content_box(h_in, w_in, out_h, out_w, px_per_token)` returning the content slice in token coordinates — computed from the same `scale=min` geometry as `resize_letterbox_pad`, so the two cannot drift.
2. `src/openpi_vega3d/towers/wan_t2v_encoder.py`: new ctor kwarg `content_region_pool: bool = False` (flows through the existing `generative_vision_tower_*` config pattern). When set, **both** `_forward_window_batch` and `_forward_single_video` slice `feats[..., top:bottom, left:right]` before `adaptive_avg_pool2d`. Same slice in both paths — this is NOT eval-path unification (Break 3, deferred).
3. `src/openpi_vega3d/towers/wan_tower.py`: kwarg pass-through.
4. `scripts/precompute_tower_features.py`: fold a `_cpool` marker into `variant_tag` when `content_region_pool=True` (mirror the existing `text_tag` pattern at line 361-362), and record the flag in the cache's `meta.json`. **Without this, the Phase 9 regen silently no-ops:** the local dir comes from the config (`:274` — the `_cpool` cache dir works), but the **S3 prefix** is auto-derived from `variant_tag` (`:362-363`), which today carries no pooling marker → the regen would hit the *legacy* S3 prefix, `list_completed_episodes_s3` (`:367`) would report every episode already done, and nothing would be computed (or, with `--s3_prefix` forced, the legacy before-evidence would be overwritten). The code's own comment (`:355-357`) requires geometry knobs to be baked into the prefix; `content_region_pool` is a new geometry knob.
5. `scripts/sync_tower_features_to_s3.sh`: parameterize the hardcoded `LOCAL_DIR` (line 6 pins the legacy `wan_t2v_16x1536_w1s1_blk20` path). It cannot clobber the legacy prefix (it syncs legacy→legacy), but as-is it would silently never upload the new `_cpool` cache.

Read-side safety (why 4–5 complete the fix): training loads features from the **explicit** `config.data.tower_features_cache_dir` — it never recomputes a prefix, so it cannot accidentally read the legacy cache once the fidelityfix config points at the `_cpool` dir.

### 8.2 — Break-2a fix: normed blend

**Reason:** A convex blend is only meaningful when both streams share a magnitude scale.
**Source:** Audit Break 2. Our code: `src/openpi/models/adaptive_gated_fusion.py:64-89` (JAX) and `src/openpi/models_pytorch/adaptive_gated_fusion.py:70-73` (torch — whose docstring at lines 9-12 already *claims* "LayerNorm on each stream resolves the scale mismatch"; the LN never reaches the blend). Reference: VEGA `feature_fusion.py:217-221` (byte-verified) blends the post-LN variables `f2d`/`fgen`. Literature: SD-DINO (2305.15347) L2-normalizes both streams explicitly "to align their scales and distributions"; Flamingo (2204.14198) and ControlNet (2302.05543) zero-init-gate injected streams; **no surveyed work convex-blends raw un-normalized heterogeneous streams**. Note the provenance: VEGA's paper Eq. 8 blends the *pre-LN projected* features — the paper and the deployed code disagree, our port matched the paper, and VEGA's published numbers came from the code.
**Problem:** LN feeds only the gate; the blend combines RAW streams. Raw block-20 DiT residual activations and SigLIP embeddings have no reason to share magnitude, so the louder stream dominates at any gate value — a "g=0.5" mix can be effectively 90/10.
**What the fix does:** Blends the LayerNormed streams: `out = (1-g)·LN(f_gen) + g·LN(f_sem)`. New flag `blend_normed: bool` on both fusion modules (JAX and torch), wired from a new `Pi0Config.vega3d_blend_normed: bool = False`. Flag off = legacy raw blend, bit-identical (Locked Decisions 3 and 6). Docstrings corrected in both files. Model-side only — no cache impact.

### 8.3 — Break-2b fix: P_gen → mlp2x_gelu

**Reason:** The generative stream needs enough projector capacity to map DiT residual space into PaliGemma token space.
**Source:** Audit Break 2. Our code: `src/openpi/models/pi0.py:141` — single `nnx.Linear(1536→2048)`; torch twin in `src/openpi/models_pytorch/pi0_pytorch.py`. Reference: VEGA deploys `mlp2x_gelu` (Linear→GELU→Linear, `multimodal_projector/builder.py`) on **both** streams — `train_wan_t2v_online.sh:68,111` (byte-verified); paper Eq. 6 projects both streams. Literature: LLaVA-1.5 (2310.03744) measured the linear→2-layer-MLP connector upgrade (+158.2 MME from that change alone); REPA (2410.06940) — the canonical DiT-residual→external-encoder mapping — required a 3-layer MLP + cosine objective, citing the "significant semantic gap" of raw diffusion hidden states. No precedent for a single Linear sufficing from a DiT residual stream.
**Problem:** A single linear map is below the demonstrated capacity floor for this source distribution.
**What the fix does:** New flag `Pi0Config.vega3d_p_gen_mlp: bool = False`. When set, `P_gen` becomes `Linear(1536→2048) → GELU → Linear(2048→2048)` (exact mlp2x_gelu shape) in both `pi0.py` and `pi0_pytorch.py`. The weight-loader backfill regex `(P_gen|P_sem|fusion|...)/.*` (`weight_loaders.py:56`) already covers any nested `P_gen` structure — merge verified in 8.5. Default nnx init (Locked Decision 5). Model-side only.

### 8.4 — Fidelity-fix training config

**Reason:** One config that turns on all three fixes, pointed at a fresh cache namespace, so Phase 9 is a two-command affair.
**Source:** Headline config `pi05_libero_fft_wan_precomp_gatewarmup` (`config.py:1155-1204`) — the config behind the published 35.3%.
**Problem:** Existing configs must stay byte-identical to their caches (Locked Decision 3); the fixes need a new config and a new cache directory as a regen marker.
**What the fix does:** Adds `pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix` — a clone of the headline config with exactly four deltas:
- `vega3d_tower_kwargs += {"content_region_pool": True}`
- `vega3d_blend_normed=True`
- `vega3d_p_gen_mlp=True`
- `tower_features_cache_dir=.../wan_t2v_16x1536_w1s1_blk20_cpool`

Everything else identical (b64, peak 1e-5, 30k steps, gate warmup 4k → 0.2, same cameras, same val split) for a clean A/B against 35.3%.

The `_cpool` dir name must match what precompute's updated `variant_tag` generates (8.1 change #4) — local dir and S3 prefix then agree by construction.

### 8.5 — Tests

**Reason:** The breaks were geometry/scale bugs that type-checked fine; the tests must check geometry and scale, not shapes.
**Source:** Existing patterns `scripts/test_tower.py`, `src/openpi/models/pi0_test.py`.
**Problem:** No current test would catch either break (both produce well-formed tensors of the right shape).
**What the fix does:** Five tests:
1. **Synthetic-pillarbox geometry** — targets `letterbox_content_box()` plus a standalone slice-and-pool unit (pure tensor math, **no WAN checkpoint** → runs in CI; the repo's CI runs `uv run pytest -m "not manual"`). Feed a token grid that is pad-valued except a known bright quadrant; with the content slice applied, assert pooled energy lands in the corresponding quadrant of the 16×16 grid and no output column is pad-dominated. A full-encoder variant (real WAN forward) goes behind `-m manual` for the remote.
2. **Normed-blend scale** — two streams with norms differing ×100; assert flag-on fused output is scale-bounded, flag-off reproduces legacy values exactly.
3. **P_gen MLP merge** — flag on → `CheckpointWeightLoader` backfills the MLP params via the existing regex; output shape unchanged `(B, 256, 2048)`.
4. **Legacy regression** — all three flags off → outputs identical to HEAD `747a5c1` behavior.
5. **JAX↔torch fusion parity** — identical weights and inputs into `models/adaptive_gated_fusion.py` and `models_pytorch/adaptive_gated_fusion.py` → matching outputs, flag on AND off (CPU-only, checkpoint-free). Today the torch twin has **zero** test coverage and no parity test exists anywhere; the blend fix lands in both frameworks, and framework divergence is a demonstrated failure mode in this codebase (see the 8.6 known-issues entry on torch P_sem).

### 8.6 — Documentation

**Reason:** Repo convention — every phase lands `CHANGELOG.md` (newest-first) and `TEST_STATUS.md` entries.
**Source:** `docs/CHANGELOG.md` (incl. the known-issues item "Tower `output_spatial` defaults still 14"), `docs/TEST_STATUS.md`.
**Problem:** Without the entries, the audit provenance and the flag semantics live only in the research wiki.
**What the fix does:** Phase 8 entries in both docs summarizing the audit provenance, the three flags, the new config, and the deferred-Break-3 note. Drive-by: since `wan_t2v_encoder.py` and `wan_tower.py` are already being touched, flip their `output_spatial` defaults 14→16 per the CHANGELOG's own known-issues recommendation, and note it in the entry.

Two findings from the 2026-06-07 review are recorded as **known-issues** (scope kept strictly to Breaks 1–2 per user decision; both deferred, not fixed):
- **Torch twin creates P_sem unconditionally** — `pi0_pytorch.py:145` always builds `self.P_sem = nn.Linear(hidden, hidden)` and applies it at `:274-276`; the JAX side respects `vega3d_use_p_sem=False` (the headline setting). NOT in the published eval path — serving auto-detects the framework (`policy_config.py:48-54`) and the published checkpoints are JAX. Parity nit; fix is one conditional whenever the torch constructor is next touched.
- **`output_spatial` default flip covers 2 of the CHANGELOG item's 4 files** — the VAE towers (`vae_online_encoder.py:31`, `vae_tower.py:29`) still default 14; they are unrelated to WAN fusion, so they stay out of this branch. The CHANGELOG known-issues item remains half-open.

---

## Files Modified (Complete List)

| File | Change | Sub-phase |
|------|--------|-----------|
| `scripts/diagnose_wan_fidelity.py` | New: column-energy + norm-ratio diagnostics | 8.0 |
| `src/openpi_vega3d/towers/common.py` | `letterbox_content_box()` helper | 8.1 |
| `src/openpi_vega3d/towers/wan_t2v_encoder.py` | `content_region_pool` kwarg; content slice before pool in both forward paths; output_spatial default 14→16 | 8.1, 8.6 |
| `src/openpi_vega3d/towers/wan_tower.py` | kwarg pass-through; output_spatial default 14→16 | 8.1, 8.6 |
| `scripts/precompute_tower_features.py` | `_cpool` marker in `variant_tag`; `content_region_pool` in meta.json | 8.1 |
| `scripts/sync_tower_features_to_s3.sh` | parameterize hardcoded legacy `LOCAL_DIR` | 8.1 |
| `src/openpi/models/adaptive_gated_fusion.py` | `blend_normed` flag; docstring correction | 8.2 |
| `src/openpi/models_pytorch/adaptive_gated_fusion.py` | `blend_normed` flag; docstring correction | 8.2 |
| `src/openpi/models/pi0_config.py` | `vega3d_blend_normed`, `vega3d_p_gen_mlp` fields | 8.2, 8.3 |
| `src/openpi/models/pi0.py` | P_gen MLP construction; flags wired to fusion | 8.2, 8.3 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | same, torch side | 8.2, 8.3 |
| `src/openpi/training/config.py` | `pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix` | 8.4 |
| `scripts/test_tower.py` | pillarbox geometry test | 8.5 |
| `src/openpi/models/pi0_test.py` | blend/merge/legacy tests | 8.5 |
| `docs/CHANGELOG.md` | Phase 8 entries | 8.6 |
| `docs/TEST_STATUS.md` | Phase 8 test tables | 8.6 |

---

## How to Use (After Phase 8 — execution is Phase 9)

```bash
# Diagnostics (remote, against the EXISTING cache/checkpoint — run BEFORE regen for the before/after)
python scripts/diagnose_wan_fidelity.py column-energy \
    --cache_dir tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20 --episodes 5
python scripts/diagnose_wan_fidelity.py norm-ratio \
    --config pi05_libero_fft_wan_precomp_gatewarmup --checkpoint <ckpt>

# Regenerate the cache with the fix (Phase 9)
# Lands in the NEW _cpool local dir (from the config) and the NEW _cpool S3 prefix
# (from the updated variant_tag) — the legacy cache/prefix is untouched and stays
# available as the before-evidence.
python scripts/precompute_tower_features.py pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix \
    --window 1 --s3_bucket behavior-challenge

# Fidelity-fixed training run (Phase 9). NOTE: train.py takes the config name
# POSITIONALLY (tyro), not via --config; --exp-name is required.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix --exp-name=fidelityfix_v1 --overwrite

# Or use the staged runbook: scripts/phase9.sh  (see that file's header)
```

**Phase 9 decision rule (the discriminating experiment, for reference):** eval the 4 swap suites at 50 episodes/task. Result ≤ 42% (no better than FFT alone) → "genuinely not useful here" is earned; the paper's §5.3/§6.2 interpretation stands fully. Result meaningfully > 35.3% → the fidelity story was real; revisit the tower verdicts.

---

## Verification (Acceptance Criteria)

1. Synthetic-pillarbox test passes: with `content_region_pool=True`, zero pad contribution to pooled tokens; quadrant geometry maps correctly to the 16×16 grid (CI-safe: helper + standalone pool unit, no checkpoint)
2. Normed-blend test passes: ×100 stream-norm imbalance → flag-on fused output scale-bounded; flag-off bit-identical to legacy
3. Legacy regression: with all three flags off, every existing config produces outputs identical to HEAD `747a5c1` behavior
4. All existing configs still parse; the new fidelityfix config parses and round-trips its flags
5. `CheckpointWeightLoader` backfills the MLP `P_gen` via the existing regex — no key errors, shapes merge cleanly
6. `probe_wan.py` output contract unchanged: `(1, 256, 1536)`
7. `diagnose_wan_fidelity.py` runs against a small synthetic cache (CI-safe smoke test, no remote needed)
8. JAX↔torch fusion parity: identical weights/inputs → matching outputs, `blend_normed` on and off
9. Cache-namespace separation: with `content_region_pool=True`, precompute's `variant_tag` (and hence the default S3 prefix) carries the `_cpool` marker — a regen pointed at a populated legacy S3 prefix must NOT report episodes already complete, and `meta.json` records the flag
10. `ruff check . && ruff format .` passes
