# Phase 4 Investigation — Ground Truth for Training Setup

**Date:** 2026-04-21
**Scope:** Sub-Phase 4.0 — answer the open questions before writing any training code.

---

## 1. Training Infrastructure (Sub-Phase 4.0.a)

### What exists in openpi-Vega3D

| Asset | Path | Status |
|-------|------|--------|
| PyTorch trainer | `scripts/train_pytorch.py` | ✅ Exists. DDP-aware, supports gradient checkpointing, bf16 precision, CheckpointWeightLoader. |
| Training config schema | `src/openpi/training/config.py` (lines 466-557 `TrainConfig`) | ✅ Exists. Supports `freeze_filter`, `optimizer`, `lr_schedule`, `pytorch_training_precision`. |
| Flow-matching loss | `src/openpi/models_pytorch/pi0_pytorch.py` (line 452, `forward()`) | ✅ Computes `F.mse_loss(u_t, v_t, reduction="none")`. Returns `[B, T, A]`; train loop aggregates via `.mean()`. |
| Data loader | `src/openpi/training/data_loader.py` (line 130) | ✅ Uses HuggingFace `datasets.LeRobotDataset`. DDP-aware batch-splitting. |
| Task embeddings in forward | `src/openpi/models_pytorch/pi0_pytorch.py` (line 280-284) | ✅ Plumbed, takes `task_id` via `_preprocess_observation`. |
| Gradient checkpointing | `pi0_pytorch.py:163-191` | ✅ Implemented for PaliGemma LM + vision tower + Gemma expert. |

### What's missing (blockers)

1. **No `LeRobotB1KDataConfig` in openpi-Vega3D.** It **exists upstream** at `/workspace/openpi/src/openpi/training/config.py:407` — needs to be ported (copy + adjust paths).
2. **No B1K TrainConfig entry.** Upstream has 9 B1K entries (lines 746-1192). We'll add one for VEGA-3D-enabled training, based on upstream's `pi0_b1k` (line 770).
3. **No adapter-only optimizer filter.** `train_pytorch.py:458-464` passes `model.parameters()` wholesale to AdamW. Need to filter to `P_gen`, `P_sem`, `fusion.*` before passing.
4. **No `torch.autocast` block in the train loop.** Current setup relies on model-internal dtype conversions. Sufficient for Phase 3 inference but should verify under training gradients.
5. **No gate-stats logging.** Phase 4 needs this as the primary diagnostic for whether fusion is learning (gate collapse → fusion learned nothing).

---

## 2. Training Data (Sub-Phase 4.0.b)

### What's on disk

| Path | Contents | Size |
|------|----------|------|
| `/workspace/BEHAVIOR-1K/datasets/2025-challenge-task-instances/metadata/` | `episodes.jsonl` (10K episodes), `B50_task_misc.csv` (50 task names) | 400 MB — **metadata only** |
| `/workspace/openpi/outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos/norm_stats.json` | Normalization statistics (mean/std/q01/q99 for state & actions) | Small, already symlinked into openpi-Vega3D |
| `/workspace/RLinf/safetensors_ckpts/openpi_05_20251115_050323_9000_tor/` | Pretrained B1K checkpoint (used for Phase 2/3 rollouts) | ~7 GB |

### What's NOT on disk (the data that matters)

The actual demo trajectories live on **HuggingFace Hub** at `behavior-1k/2025-challenge-demos`:
- 10,000 `.parquet` files (one per episode, actions/state/proprio)
- 69,339 `.mp4` files (3 cameras × ~23K — some episodes have extra angles)
- 3 `.jsonl` files (task descriptions, skill annotations)
- Full repo total size: estimated ~700 GB for 10K episodes; ~14 GB for the 200-episode subset we actually need.

### Data format

LeRobot dataset format (HuggingFace). Per-step keys that reach `B1kInputs`:

| B1K expected key | Comes from LeRobot | Verified |
|------------------|---------------------|----------|
| `observation/egocentric_camera` | `observation.images.rgb.head` | ✅ |
| `observation/wrist_image_left` | `observation.images.rgb.left_wrist` | ✅ |
| `observation/wrist_image_right` | `observation.images.rgb.right_wrist` | ✅ |
| `observation/state` | `observation.state` (256-dim) | ✅ |
| `actions` | `action` (23-dim in LeRobot, padded to 32 in model) | ✅ |
| `task_index` | `task_index` (0-49) | ✅ |
| `prompt` | language instruction | ✅ |

### Key upstream pattern to borrow

Upstream trains on only **190 episodes** (not all 10K) with a 10-episode val set:
```python
episodes_index=list(range(190)),           # train
val_episodes_index=list(range(190, 200)),  # val
```
This keeps downloads, compute, and iteration time manageable. We'll mirror this.

### Disk budget

- Free space on `/workspace`: **335 GB**
- Estimated subset download: ~14 GB (200 episodes × ~70 MB for 3 videos + parquet)
- Comfortable headroom.

---

## 3. Evaluation Tasks (Sub-Phase 4.0.c)

### Criteria

- **At least one verified-working task:** `turning_on_radio` (Phase 2/3 smoke tests).
- **A mix of difficulty** so the learned-gate vs baseline delta is detectable:
  - *Easy (simple manipulation):* `turning_on_radio`, `picking_up_trash`
  - *Medium (pick-and-place):* `putting_away_Halloween_decorations`, `carrying_in_groceries`
  - *Hard (multi-step):* `cleaning_up_plates_and_food`, `chop_an_onion`

### Proposed eval set (5 tasks)

1. `turning_on_radio` — our known-good smoke task
2. `picking_up_trash` — simple manipulation
3. `putting_away_Halloween_decorations` — pick-and-place
4. `carrying_in_groceries` — navigation + manipulation
5. `chop_an_onion` — tool use / precision

### Rollout count per task

Recommend **20 rollouts per task** (total 5 × 20 × 3 methods = 300 rollouts). 10 is too noisy to distinguish learned-gate vs baseline; 50 is overkill for a first-pass eval.

---

## 4. GPU Budget (Sub-Phase 4.0.d)

| Metric | Value |
|--------|-------|
| GPU | NVIDIA RTX 6000 Ada Generation |
| Total VRAM | 48 GB |
| Free at start of Phase 4 | 48.5 GB |
| Compute capability | 8.9 (native bf16 support) |
| Driver | 570.211.01 |

### Memory model estimate

| Component | Params | bf16 memory | Gradient | Optimizer (AdamW ×2) | Total |
|-----------|--------|-------------|----------|----------------------|-------|
| PaliGemma + expert (frozen) | ~3.5B | 7 GB | 0 | 0 | 7 GB |
| SD2.1 VAE tower (frozen) | 80M | 160 MB | 0 | 0 | 160 MB |
| `P_gen` + `P_sem` + `fusion` (trainable) | ~12M | 24 MB | 24 MB | 48 MB | 96 MB |
| Activations (batch=8, bf16) | — | ~10-15 GB | — | — | ~12 GB |
| **Subtotal** | — | — | — | — | **~20 GB** |

With gradient checkpointing enabled (already in code), activation memory drops ~40%. Budget is comfortable for batch size 8; batch 16 is probably fine too.

**Conclusion:** single-GPU training is feasible. No DDP or model-parallel needed.

---

## 5. Key Design Decisions (Locked for Phase 4)

1. **Train on the upstream's 190-episode subset.** Matches their baseline; faster iteration; comparable results.
2. **Single-GPU training** (no DDP for Phase 4.0).
3. **Adapter-only training:** freeze everything except `P_gen`, `P_sem`, `fusion.*` (including the LayerNorms inside fusion).
4. **VAE tower first** (not WAN). WAN is an optional Phase 5.
5. **Training-time spatial dropout = 0.1** (from Phase 3 plan).
6. **Evaluation: 5 tasks × 20 rollouts × 3 methods (baseline vs `force_gate=1.0` vs learned).**

---

## 6. Revised Phase 4 Sub-Phase Plan

Based on investigation, the path forward is:

### 4.1 — Port data config + create B1K-VEGA3D TrainConfig
- Port `LeRobotB1KDataConfig` from upstream (`openpi/src/openpi/training/config.py:407`).
- Add new `TrainConfig` entry `pi05_b1k_vega3d` (or similar) with `use_vega3d=True`, `vega3d_tower_kwargs={"checkpoint_dir": ...}`.
- Verify: config parses, data loader constructs (with network on, will trigger HF download of the 190 episodes).

### 4.2 — Download training data subset
- Trigger HF download of just episodes 0-199 from `behavior-1k/2025-challenge-demos`.
- Verify: files land on disk, size is in estimate range (~14 GB).

### 4.3 — Adapter-only freeze + training script patches
- In `train_pytorch.py`, filter `model.parameters()` to only `P_gen`, `P_sem`, `fusion.*` before passing to AdamW.
- Add param-count sanity check at start: should be ~12M trainable out of 3.5B total.
- Add gate-stats logging (mean, std, histogram of `g` per step).

### 4.4 — Fix `to_unit_range().item()` compile fallback
- Replace with `torch.aminmax` or fixed SD2.1 normalization constants.
- Verify: `torch.compile` no longer falls back to eager at tower input.

### 4.5 — Training-time spatial dropout
- During `forward()` training pass, with `p=0.1`, force `force_gate=1.0` for the step (zero out tower contribution for that batch).
- Verify: dropout path triggers stochastically; no exception under either branch.

### 4.6 — First training run (smoke, 100-500 steps)
- Goal: confirm loss decreases, gate doesn't collapse, no OOM.
- Log: `loss`, `g_mean`, `g_std`, trainable-param norms, wall clock per step.
- Save checkpoint.

### 4.7 — Convergence training
- Run 10K-50K steps on the 190-episode subset.
- Monitor val loss at intervals.
- Save final checkpoint.

### 4.8 — Evaluation
- Run 3 method conditions × 5 tasks × 20 rollouts.
- Record success rate, mean reward, wall time.
- Compare: learned gate vs `force_gate=1.0` vs baseline.
- Write results to `docs/PHASE4_RESULTS.md`.

---

## 7. Risks Remaining

| Risk | Mitigation |
|------|-----------|
| **Gate collapse** during training (g → 1 or 0 uniformly) | Log per-step; add entropy regularization if observed (`L -= λ · H(g)`, small λ). |
| **Data download interrupted** | Use HF `snapshot_download` with resume-friendly flags; verify integrity post-download. |
| **Memory creeping past 48GB** under training (vs ~20GB est.) | Gradient checkpointing already on; can reduce batch size; can offload optimizer state. |
| **torch.compile breakage elsewhere** | Keep `TORCHDYNAMO_DISABLE=1` option available as escape hatch. |
| **LeRobot dataset schema drift** between upstream's version and current HF repo | Verify one sample loads correctly before starting long training. |

---

## 8. Answering the Overnight Decisions

From Phase 4 game plan:

1. **Single or multi-task?** → **Multi-task.** Task embeddings are plumbed; 50 tasks × 190 episodes is rich enough to exercise them.
2. **How many eval rollouts?** → **20 per task × 5 tasks × 3 methods = 300 total.** Solid without being rigorous-overkill.
3. **Fix `to_unit_range` now?** → **Yes, Sub-Phase 4.4.** Speeds training throughput meaningfully.
4. **Spatial dropout?** → **Yes, p=0.1, Sub-Phase 4.5.** Near-zero cost, clean ablation.

---

## 9. Ready State

All four investigation questions have answers. No hidden blockers. Phase 4 sub-phases are concrete and ordered. Next step is **Sub-Phase 4.1 — port the B1K data config + create the TrainConfig entry.**
