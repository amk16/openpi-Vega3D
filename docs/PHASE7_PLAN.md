# Phase 7: DreamDojo Training Setup — Full Plan

**Last updated:** 2026-05-20
**Status:** Complete. All sub-phases 7.0–7.6 done.

---

## What Phase 7 Is

Phase 7 brings DreamDojo to the same training-readiness as WAN on the main branch. Phase 6 built the tower infrastructure (frozen Cosmos-Predict2.5-2B encoder, registry entry, B1K configs). Phase 7 adds the training pipeline: precompute script support, LIBERO configs, and Pi0Config integration — so that adapter training can be kicked off.

**This is setup, not training.** Phase 7 delivers scripts and configs ready to run. Actual precompute execution, training runs, and evaluation are outside scope.

### Phase Numbering Rationale

| Phase | Scope | Status |
|-------|-------|--------|
| 0-3 | Infrastructure (registry, env, policy, fusion) | Done |
| 4 | VAE adapter training | In progress |
| 5 | WAN adapter training | Done on main (LIBERO) |
| 6 | DreamDojo as third backbone | Done |
| **7** | **DreamDojo training setup** | **This plan** |
| 8 (future) | DreamDojo training runs + eval | Outside this plan |

---

## Locked Decisions

| # | Decision | Value | Why |
|---|----------|-------|-----|
| 1 | Target dataset | LIBERO (not B1K) | Match WAN training for apples-to-apples comparison |
| 2 | feat_dim | 2048 (auto-derived) | 16 heads × 128 dim confirmed in Phase 6 investigation |
| 3 | Precompute mode | Single-frame (T=1) | Multi-frame windowing deferred pending Cosmos temporal attention investigation |
| 4 | Batch size (in-process) | 4 | DreamDojo 2B is ~50% larger than WAN 1.3B; 4 vs WAN's 8 |
| 5 | Batch size (precomputed) | 64 | Matches WAN precomp — no tower in GPU memory |
| 6 | Two configs (no semonly) | in-process / precomp | WAN semonly already serves as shared control — force_gate=1.0 zeros generative features regardless of tower |

---

## Sub-Phase Status

```
7.0  Merge origin/main           DONE     (training pipeline, smart init, precompute infra)
7.1  Pi0Config feat_dim           DONE     src/openpi/models/pi0_config.py
7.2  Probe script                 DONE     scripts/probe_dreamdojo.py
7.3  Precompute adaptation        DONE     scripts/precompute_tower_features.py
7.4  LIBERO training configs      DONE     src/openpi/training/config.py (2 new configs)
7.5  B1K config fix               DONE     Commented out B1K DreamDojo configs (LeRobotB1KDataConfig disabled)
7.6  Documentation                DONE     docs/ (this file + CHANGELOG + TEST_STATUS)
```

### Dependency Graph

```
7.0  Merge origin/main → dreamdojo
 │
 ├──────────┬──────────┐
 ▼          ▼          ▼
7.1        7.2        7.3
feat_dim   probe      precompute
auto       script     adaptation
 │          │          │
 └──────────┴──────────┘
            │
            ▼
           7.4  LIBERO training configs
            │
            ▼
           7.5  B1K config fix (NameError)
            │
            ▼
           7.6  Documentation
```

---

## Detailed Changes Per Sub-Phase

### 7.0 — Merge origin/main → dreamdojo

Brought in from main:
- Precompute pipeline (`LoadPrecomputedTowerFeatures`, S3 checkpoint sync)
- Model changes (`skip_tower_construction`, P_gen zero-init, P_sem identity-init, gate bias 4.0)
- Validation data loader with held-out episodes
- `LeRobotLiberoVegaDataConfig` with `tower_features_cache_dir`
- Multi-frame windowing support (`encode_window_batch` on BaseTower)
- Three WAN LIBERO configs (in-process, precomputed, semantic-only)

One conflict resolved: `policy_utils.py` B1K import (followed main's approach — commented out, B1K path already broken on main).

### 7.1 — Pi0Config auto-derive feat_dim

Added `dreamdojo` → 2048 to `Pi0Config.__post_init__` feat_dim auto-derivation. Without this, any DreamDojo config without explicit `vega3d_tower_feat_dim` raises ValueError.

### 7.2 — Probe script

Created `scripts/probe_dreamdojo.py` mirroring `probe_wan.py`. Works in offline mode (no checkpoint needed) and online mode. Prints output shape, feat_dim, and num_tokens for config verification.

### 7.3 — Precompute script adaptation

Three changes to `scripts/precompute_tower_features.py`:
1. Added `ensure_dreamdojo_checkpoint()` — validates .pt file exists, prints download guidance if missing
2. Made `prepare_image()` accept configurable resolution (default 224 for backward compat, DreamDojo uses 256)
3. Resolved `image_resolution` from tower kwargs so DreamDojo images go directly to 256×256

### 7.4 — LIBERO training configs

Two new configs in `src/openpi/training/config.py`:

| Config | Purpose | Key settings |
|--------|---------|-------------|
| `pi05_libero_lora_dreamdojo` | In-process tower | batch=4, live tower forward |
| `pi05_libero_lora_dreamdojo_precomp` | Precomputed features | batch=64, skip_tower_construction, T=1 cache |

**No DreamDojo semonly config.** The existing `pi05_libero_lora_wan_precomp_semonly` already serves as the semantic-only control for ALL towers. With `force_gate=1.0`, generative features are multiplied by zero — the result is identical regardless of which tower produced them.

Both mirror WAN configs exactly: same LR (1e-5), same steps (30K), same optimizer (AdamW, grad clip 1.0), same val split, same freeze filter, same cameras.

### 7.5 — B1K config fix

The two B1K DreamDojo configs from Phase 6 (`pi05_b1k_dreamdojo`, `pi05_b1k_dreamdojo_wrist`) referenced `LeRobotB1KDataConfig`, which is commented out on main (along with its `b1k_policy` import). This caused a `NameError` at import time, crashing the entire config module. Both configs are now commented out with inline re-enable instructions — search "DISABLED: LeRobotB1KDataConfig" in `config.py`.

---

## Files Modified (Complete List)

| File | Change | Sub-phase |
|------|--------|-----------|
| (35 files from merge) | Merge origin/main | 7.0 |
| `src/openpi/models/pi0_config.py` | dreamdojo → 2048 feat_dim auto-derive | 7.1 |
| `scripts/probe_dreamdojo.py` | New: DreamDojo probe script | 7.2 |
| `scripts/precompute_tower_features.py` | ensure_dreamdojo_checkpoint(), configurable image resolution | 7.3 |
| `src/openpi/training/config.py` | 2 new LIBERO DreamDojo configs | 7.4 |
| `docs/PHASE7_PLAN.md` | New: this file | 7.5 |
| `src/openpi/training/config.py` | Commented out B1K DreamDojo configs (NameError fix) | 7.5 |
| `docs/CHANGELOG.md` | Phase 7 entries | 7.6 |
| `docs/TEST_STATUS.md` | Phase 7 test tables | 7.6 |

---

## How to Use (After Phase 7)

### Probe DreamDojo output shape
```bash
python scripts/probe_dreamdojo.py --checkpoint_dir ckpts/DreamDojo-2B
```

### Precompute features
```bash
python scripts/precompute_tower_features.py pi05_libero_lora_dreamdojo_precomp \
    --window 1 --s3_bucket behavior-challenge
```

### Train with precomputed features
```bash
python scripts/train.py --config pi05_libero_lora_dreamdojo_precomp
```

### Train semantic-only ablation (shared control for ALL towers)
```bash
python scripts/train.py --config pi05_libero_lora_wan_precomp_semonly
```

---

## Verification (Acceptance Criteria)

1. All WAN configs from origin/main still parse (no merge regressions)
2. DreamDojo B1K configs from Phase 6 commented out (depend on disabled `LeRobotB1KDataConfig`); re-enable instructions inline
3. Two new LIBERO DreamDojo configs parse
4. `Pi0Config` auto-derives `vega3d_tower_feat_dim=2048` for `dreamdojo`
5. `probe_dreamdojo.py` runs in offline mode without error
6. `precompute_tower_features.py` accepts dreamdojo config without WAN-specific errors
7. `TOWER_REGISTRY` contains `{"vae", "wan_t2v", "dreamdojo"}`
8. `ruff check . && ruff format .` passes
