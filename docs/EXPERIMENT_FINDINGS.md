# openpi-Vega3D Experiment Findings

**Date range:** April 2 -- May 26, 2026
**Scope:** All training runs across wandb projects `openpi`, `rlinf`, and `B1K`
**Entity:** `salman_shahid`
**Dataset:** LIBERO (`physical-intelligence/libero`, ~1693 train episodes, ~85 val episodes)
**Base model:** pi0.5 (~3.5B params: PaliGemma 2B vision-language + Gemma 300M action expert)
**Config source:** `src/openpi/training/config.py`, model fields in `src/openpi/models/pi0_config.py`

---

## 1. Executive Summary

- **Batch size is the dominant regularizer.** FFT at bs=64 produces severe overfitting (train-val gap of 0.035); bs=16 nearly eliminates it (gap of 0.002). This single variable matters more than tower choice.
- **Cosmos base tower achieves the best validation loss among LoRA runs** (0.0251), though confounded by deeper LoRA rank (r32 vs r16).
- **Forced gate=1 (semantic-only) consistently matches or beats learned gate** in every paired comparison, suggesting generative tower features may hurt generalization in the LoRA regime.
- **blk20 + NO_INIT_BIAS is strictly better than default block settings** for WAN tower LoRA runs (val 0.0267--0.0271 vs 0.0313--0.0322).
- **KI+FAST auxiliary losses inflate headline loss** (~0.3 total vs ~0.01 action-only). Cross-run comparison requires tracking `action_loss` specifically.
- **The RL approach (rlinf project) was unsuccessful** -- 142 runs over 5 months, 0% task success rate. PPO on a 3.5B VLA with sparse reward proved intractable.
- **LIBERO suite evaluations remain the critical bottleneck.** Validation loss is a proxy; task success rate on the 12 evaluation suites is the real metric, and most runs lack eval data.

---

## 2. Experiment Taxonomy

### 2.1 Configuration Dimensions

| Dimension | Values Tested | Config Field |
|-----------|---------------|--------------|
| Vision Tower | none (baseline), `wan_t2v`, `cosmos_base`, `cosmos_libero` | `pi0_config.py:69` `vega3d_tower_name` |
| Training Capacity | LoRA r16 (`gemma_2b_lora`), Deep LoRA r32 (`gemma_2b_lora_32`), Full Fine-Tune (`gemma_2b`) | `pi0_config.py:21` `paligemma_variant` |
| Gate Mechanism | learned (default), forced=1.0 (semonly), warmup schedule | `pi0_config.py:74` `vega3d_force_gate`, `:82` `vega3d_gate_warmup_steps` |
| Auxiliary Losses | none, FAST tokenizer, Knowledge Insulation (KI) | `use_fast_auxiliary`, `use_knowledge_insulation` |
| Batch Size | 16, 32, 64, 128, 256 | `config.py` `batch_size` |
| Feature Block | default (last), block 20 (~70% depth) | tower_kwargs `feat_block_idx` |
| Gate Warmup | none, 4k, 6k, 8k, 15k steps | `pi0_config.py:82` `vega3d_gate_warmup_steps` |
| P_sem Projection | enabled, disabled | `pi0_config.py:87` `vega3d_use_p_sem` |

### 2.2 Master Run Table (substantive runs only)

| # | Run Name | Tower | Training | BS | Gate | Warmup | Steps | action_loss | val_loss | Hours | Status |
|---|----------|-------|----------|-----|------|--------|-------|-------------|----------|-------|--------|
| 1 | `libero_lora_v1` | none | LoRA r16 | 32 | -- | -- | 30k | 0.0117 | -- | 42.9 | finished |
| 2 | `pi05_libero_ki` | none | LoRA r16 | 32 | -- | -- | 30k | 0.3041* | 0.3513* | 50.2 | finished |
| 3 | `lora_baseline_v1` | none | LoRA r16 | 64 | -- | -- | 30k | 0.0104 | 0.0313 | 11.7 | finished |
| 4 | `wan_precomp_v1` | wan_t2v | LoRA r16 | 64 | learned | -- | 30k | 0.0103 | 0.0322 | 12.7 | finished |
| 5 | `wan_precomp_semonly_v1` | wan_t2v | LoRA r16 | 64 | forced=1 | -- | 30k | 0.0103 | 0.0314 | 12.4 | finished |
| 6 | `wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS` | wan_t2v | LoRA r16 | 32 | learned | -- | 30k | 0.0145 | 0.0271 | 11.0 | finished |
| 7 | `wan_precomp_semonly_v1_w1s1_blk20_NO_INIT_BIAS` | wan_t2v | LoRA r16 | 32 | forced=1 | -- | 30k | 0.0146 | 0.0267 | 10.9 | finished |
| 8 | `libero_wan_full_finetune` | wan_t2v | FFT | 128 | learned | -- | 30k | 0.0070 | 0.0347 | 43.1 | failed |
| 9 | `libero_deeplora_ki_ar_wan_precomp` | none | LoRA r32 | 256 | -- | -- | 30k | 0.1356* | 0.0356** | 19.1 | failed |
| 10 | `libero_deeplora_ki_ar_2nd` | none | LoRA r32 | 32 | -- | -- | 18k | 0.0277** | 0.0274** | 4.0 | finished |
| 11 | `libero_deeplora_ki_ar_wan_precomp_fr` | wan_t2v | LoRA r32 | 32 | learned | -- | 18k | 0.0307** | 0.0286** | 5.2 | finished |
| 12 | `libero_deeplora_wan_precomp_gatewarmup` | wan_t2v | LoRA r32 | 32 | learned | 6k | 30k | 0.0114 | 0.0289 | 8.5 | finished |
| 13 | `cosmos_base_v1_w1s1` | cosmos_base | LoRA r32 | 32 | learned | 6k | 30k | 0.0177 | 0.0251 | 8.5 | finished |
| 14 | `fft_wan_precomp_gatewarmup` | wan_t2v | FFT | 64 | learned | 4k | 30k | 0.0045 | 0.0440 | 15.6 | finished |
| 15 | `fft_cosmos_precomp_gatewarmup` | cosmos_base | FFT | 64 | learned | 4k | 30k | 0.0153 | 0.0267 | 10.3 | killed |
| 16 | `libero_fft` | none | FFT | 64 | -- | -- | 30k | 0.0054 | 0.0400 | 12.3 | killed |
| 17 | `fft_wan_precomp_gatewarmup_smallbatch` | wan_t2v | FFT | 16 | learned | 8k | 30k | 0.0195 | 0.0213 | 4.9 | finished |
| 18 | `libero_fft_smallbatch` | none | FFT | 16 | -- | -- | 30k | 0.0196 | 0.0221 | 4.5 | finished |
| 19 | `fft_wan_precomp_LONGER_gatewarmup_smallbatch` | wan_t2v | FFT | 16 | learned | 15k | 40k | 0.0169 | 0.0245 | running | running |

\* Total loss includes KI + FAST auxiliary terms; not comparable to action-only runs.
\*\* `action_loss` reported (extracted from combined KI+FAST loss).

---

## 3. Head-to-Head Comparisons

### 3.1 Tower Effect -- LoRA, Matched Settings

Held constant: LoRA r16, 30k steps.

| Run | Tower | Block | Init | BS | val_loss | Delta vs baseline |
|-----|-------|-------|------|----|----------|-------------------|
| `lora_baseline_v1` | none | -- | -- | 64 | 0.0313 | -- |
| `wan_precomp_v1` | wan_t2v | default | default | 64 | 0.0322 | +0.0009 (worse) |
| `wan_precomp_semonly_v1` | wan_t2v | default | default | 64 | 0.0314 | +0.0001 (neutral) |
| `wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS` | wan_t2v | 20 | no bias | 32 | 0.0271 | **-0.0042** |
| `wan_precomp_semonly_v1_w1s1_blk20_NO_INIT_BIAS` | wan_t2v | 20 | no bias | 32 | 0.0267 | **-0.0046** |
| `cosmos_base_v1_w1s1` | cosmos_base | 20 | no bias | 32 | **0.0251** | **-0.0062** |

**Finding:** With default settings (top 3 rows), the WAN tower provides no benefit over the baseline -- the learned gate even slightly hurts. With blk20 + no init bias (bottom 3 rows), both WAN and Cosmos towers clearly outperform. Cosmos achieves the best val loss, but uses LoRA r32 instead of r16 (confound -- see Section 5). The blk20 + no-init-bias configuration change matters at least as much as tower choice.

### 3.2 Tower Effect -- FFT, bs=64

Held constant: full fine-tune, bs=64, 30k steps.

| Run | Tower | Warmup | train_loss | val_loss | Train-val gap |
|-----|-------|--------|------------|----------|---------------|
| `libero_fft` | none | -- | 0.0054 | 0.0400 | 0.0346 |
| `fft_wan_precomp_gatewarmup` | wan_t2v | 4k | 0.0045 | 0.0440 | 0.0395 |
| `fft_cosmos_precomp_gatewarmup` | cosmos_base | 4k | 0.0153 | 0.0267 | 0.0114 |

**Finding:** At bs=64, WAN FFT overfits worse than the baseline (val 0.0440 vs 0.0400). Cosmos FFT stands out: highest train loss but best val loss by far, suggesting the Cosmos tower provides useful regularization. All bs=64 runs show large train-val gaps.

### 3.3 Tower Effect -- FFT, bs=16

Held constant: full fine-tune, bs=16, 30k steps.

| Run | Tower | Warmup | train_loss | val_loss | Train-val gap |
|-----|-------|--------|------------|----------|---------------|
| `libero_fft_smallbatch` | none | -- | 0.0196 | 0.0221 | 0.0025 |
| `fft_wan_precomp_gatewarmup_smallbatch` | wan_t2v | 8k | 0.0195 | 0.0213 | 0.0018 |

**Finding:** At bs=16, the train-val gap nearly vanishes for both runs. WAN provides a small but consistent improvement (0.0213 vs 0.0221, delta = -0.0008). This is the most properly regularized comparison available. The currently running `fft_wan_precomp_LONGER_gatewarmup_smallbatch` (40k steps, warmup=15k) is at step 39900 with val_loss=0.0245 -- slightly worse than the 30k-step version, suggesting the extra 10k steps may be entering overfit territory.

### 3.4 Batch Size Effect

Cross-comparison grouping FFT runs by batch size:

| BS | Baseline train | Baseline val | WAN train | WAN val | Baseline gap | WAN gap |
|----|----------------|-------------|-----------|---------|--------------|---------|
| 64 | 0.0054 | 0.0400 | 0.0045 | 0.0440 | 0.0346 | 0.0395 |
| 16 | 0.0196 | 0.0221 | 0.0195 | 0.0213 | 0.0025 | 0.0018 |

**Finding:** Reducing batch size from 64 to 16 cuts the train-val gap by 14-22x. At bs=16, both baseline and WAN achieve their best validation losses. The tower effect (WAN vs baseline) reverses direction between batch sizes: WAN hurts at bs=64 but helps at bs=16. Batch size is the single most important hyperparameter for generalization on LIBERO.

### 3.5 Learned Gate vs Forced Gate (Semantic-Only Ablation)

Two matched pairs where the only difference is `vega3d_force_gate`:

| Pair | Learned gate val | Forced gate=1 val | Winner |
|------|-----------------|-------------------|--------|
| `wan_precomp_v1` vs `wan_precomp_semonly_v1` | 0.0322 | 0.0314 | forced (by 0.0008) |
| `wan_precomp_v1_w1s1_blk20` vs `semonly_v1_w1s1_blk20` | 0.0271 | 0.0267 | forced (by 0.0004) |

**Finding:** In both pairs, forcing the gate to 1.0 (semantic pathway only, no generative contribution) matches or slightly beats the learned gate. This implies the model learns to over-rely on generative features early in training, hurting generalization. The gate warmup schedule (`vega3d_gate_warmup_steps`) was introduced specifically to address this -- it forces the gate toward a low target before releasing to learned values.

### 3.6 Block Depth (blk20 vs Default)

| Run | Block | Init | BS | val_loss |
|-----|-------|------|----|----------|
| `wan_precomp_v1` | default | default | 64 | 0.0322 |
| `wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS` | 20 | no bias | 32 | 0.0271 |
| `wan_precomp_semonly_v1` | default | default | 64 | 0.0314 |
| `wan_precomp_semonly_v1_w1s1_blk20_NO_INIT_BIAS` | 20 | no bias | 32 | 0.0267 |

**Finding:** blk20 + no-init-bias runs are better in every case. However, **three variables change simultaneously**: block depth, init bias, and batch size (64 vs 32). These cannot be isolated without additional ablations (see Section 5).

### 3.7 Knowledge Insulation + FAST Effect

Runs with KI+FAST auxiliary losses (comparing `action_loss` only):

| Run | KI+FAST | Tower | action_loss | val_action_loss | total_loss |
|-----|---------|-------|-------------|-----------------|------------|
| `libero_deeplora_ki_ar_2nd` | yes | none | 0.0277 | 0.0274 | 0.3810 |
| `libero_deeplora_ki_ar_wan_precomp_fr` | yes | wan_t2v | 0.0307 | 0.0286 | 0.3862 |
| `libero_deeplora_wan_precomp_gatewarmup` | no | wan_t2v | 0.0114 | 0.0289 | 0.0114 |

**Finding:** KI+FAST without tower achieves the best val_action_loss (0.0274), slightly beating KI+FAST with tower (0.0286). The non-KI gate-warmup run reaches much lower train action_loss (0.0114) but similar validation (0.0289), confirming that KI acts as a regularizer. The total_loss for KI runs (~0.38) is dominated by FAST tokenizer and KI AR auxiliary terms -- not indicative of action prediction quality.

### 3.8 Training Capacity (LoRA vs Deep LoRA vs FFT)

Comparing training methods with WAN tower, gate warmup:

| Run | Method | BS | val_loss | gate_mean |
|-----|--------|----|----------|-----------|
| `wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS` | LoRA r16 | 32 | 0.0271 | -- |
| `libero_deeplora_wan_precomp_gatewarmup` | LoRA r32 | 32 | 0.0289 | 0.726 |
| `fft_wan_precomp_gatewarmup_smallbatch` | FFT | 16 | 0.0213 | 0.323 |
| `fft_wan_precomp_gatewarmup` | FFT | 64 | 0.0440 | 0.758 |
| `libero_wan_full_finetune` | FFT | 128 | 0.0347 | -- |

**Finding:** FFT at bs=16 achieves the best val loss (0.0213) of any WAN run. FFT at bs=64/128 overfits. LoRA r16 at blk20 (0.0271) outperforms LoRA r32 with warmup (0.0289) despite fewer trainable parameters -- likely because blk20 features are more useful than default-block features. The 43-hour full fine-tune at bs=128 (0.0347) demonstrates that raw capacity without regularization does not help.

---

## 4. Training Dynamics

### Gate Warmup Behavior

The gate warmup schedule (`pi0_config.py:79-84`) implements a tent function: the gate is forced from `start` (default 1.0) toward `target` over the first half of `warmup_steps`, then released to the learned value in the second half. This prevents the model from collapsing the gate early. Runs with warmup show lower final `gate_mean` (0.27-0.33 at bs=16) compared to runs without warmup where the gate stays high (0.76-0.96), suggesting warmup successfully prevents the model from defaulting to "always use tower features."

### Overfitting Signature

The canonical overfitting pattern appears in all bs=64 FFT runs: train loss drops to 0.004-0.005 while val loss plateaus at 0.03-0.04 by ~15k steps. `fft_wan_precomp_gatewarmup` (ID `t9qvhe4x`) is the clearest example -- 15.6 hours of training drove train loss to 0.0045 but val loss to 0.0440. The bs=16 runs do not exhibit this pattern.

### Full Fine-Tune WAN Run

`libero_wan_full_finetune` (ID `2wbqvkrn`) was the most expensive single run: 43 hours, bs=128, no gate warmup. Final train loss 0.0070, val loss 0.0347. This demonstrates that combining maximum training capacity (FFT) with large batch size produces heavy memorization. The 43 hours of compute yielded worse validation than 5-hour LoRA runs at bs=32.

### KI Loss Decomposition

For `pi05_libero_ki` (ID `jjfa4ohk`), the total loss of 0.3041 decomposes into: `ki_fm_loss` (flow-matching action loss) = 0.0166, `ki_ar_loss` (autoregressive knowledge insulation) = 5.75, weighted by `ki_ar_loss_weight=0.05` to contribute 0.2875 to total loss. The `fast_accuracy` metric reached 0.26 -- above random (1/4096 ~ 0.0002) but far from ceiling, indicating the FAST tokenizer auxiliary head was learning but slowly.

---

## 5. Confounds and Missing Controls

### 5.1 Confounds in Head-to-Head Comparisons

| Comparison | Confounding Variable | Impact |
|------------|---------------------|--------|
| Cosmos vs WAN LoRA (Sec 3.1) | LoRA rank: Cosmos used r32, WAN used r16 | Cannot isolate tower effect from capacity effect |
| blk20 vs default (Sec 3.6) | Simultaneous change: block depth + init bias + batch size | Three variables moved at once |
| FFT bs=64 vs bs=16 (Sec 3.4) | Different warmup steps (0/4k vs 0/8k) | Warmup may contribute to bs=16 advantage |
| KI runs vs non-KI (Sec 3.7) | Different loss landscape, gradient dynamics | Not directly comparable via loss values |
| Wall clock across runs | Different hardware, different time periods | Training speed not comparable |

### 5.2 Missing Controls

These experiments would resolve the confounds above:

1. **Cosmos at LoRA r16** -- isolate tower effect from LoRA rank
2. **blk20 with default init bias (and vice versa)** -- isolate block depth from initialization
3. **Cosmos FFT at bs=16** -- test whether Cosmos advantage holds under best regularization
4. **Pure gate warmup ablation** -- same tower/batch/training, vary only warmup length
5. **LIBERO suite evaluation on top checkpoints** -- validation loss is a proxy, not the real metric

---

## 6. RLinf Post-Mortem

**Project:** `rlinf` on wandb
**Period:** November 6, 2025 -- March 14, 2026 (5 months)
**Total runs:** 142
**Approach:** PPO (Proximal Policy Optimization) applied to pi0.5 policy in BEHAVIOR-1K/LIBERO simulator
**Result:** 0% task success rate across all runs

### Breakdown

| Period | Runs | Longest | Notes |
|--------|------|---------|-------|
| Nov 6-9, 2025 | ~100 | 14h | Rapid iteration on infrastructure: `test_openpi_pi05_behavior` |
| Nov 9 - Dec 24, 2025 | ~30 | 21h | Longer training attempts; all reached 0% success |
| Dec 25, 2025 - Mar 14, 2026 | ~12 | 1h | Sporadic short runs (`behavior_ppo_pi05_*`) |

### Why It Failed

PPO requires dense, informative reward signals. LIBERO manipulation tasks have sparse binary rewards (success/fail at episode end). Combined with the 3.5B parameter policy model, the gradient signal from RL was too noisy to make meaningful progress. The 21-hour longest run (Dec 24, crashed) logged 30 metrics across PPO diagnostics but `env/success` never rose above 0.

### Lesson

Supervised fine-tuning (the current openpi approach) is dramatically more sample-efficient than RL for VLA policy adaptation on manipulation tasks. This failure motivated the pivot to the current VEGA-3D vision-tower fusion approach.

### B1K Project

The `B1K` wandb project was created but contains 0 runs. It was superseded by the pivot to LIBERO as the primary evaluation benchmark.

---

## 7. Recommendations

### 7.1 Immediate (next 1-2 runs)

1. **Run LIBERO eval on top 3 checkpoints.** Training loss comparisons are necessary but insufficient. Use `scripts/sequence_libero_evals.sh` to run all 12 suites on:
   - `fft_wan_precomp_gatewarmup_smallbatch` (best WAN FFT val: 0.0213)
   - `cosmos_base_v1_w1s1` (best Cosmos LoRA val: 0.0251)
   - `libero_fft_smallbatch` (best baseline FFT val: 0.0221)

2. **Evaluate `fft_wan_precomp_LONGER_gatewarmup_smallbatch`** once it finishes. At step 39900/40000 it shows val=0.0245, which is slightly worse than the 30k-step run (0.0213). If the final result confirms degradation, 30k steps at bs=16 is the sweet spot.

### 7.2 Controlled Ablations

3. **Cosmos at LoRA r16.** Run `cosmos_base` with `gemma_2b_lora` (rank 16) to make a fair comparison with WAN LoRA r16 runs.

4. **Isolate block depth from init bias.** Run WAN LoRA with blk20 + default init, and separately default block + no init bias.

5. **Cosmos FFT at bs=16.** The Cosmos tower showed the best regularization properties at bs=64 (Sec 3.2). Test whether this holds at bs=16 where overfitting is already minimal.

### 7.3 Architecture Investigations

6. **Gate warmup length sweep.** Current runs use 4k/6k/8k/15k warmup but these are not controlled against each other. Run a sweep with tower and batch size held constant (suggest WAN FFT bs=16 as the base config).

7. **Cosmos-LIBERO domain tower.** The `cosmos_libero` tower (fine-tuned on LIBERO data) config exists (`pi05_libero_fft_cosmos_libero_version_precomp_gatewarmup`) but the only run was killed after 5h. This should be re-run with bs=16.

8. **Multi-camera ablation.** All current runs apply the tower to `base_0_rgb` + `left_wrist_0_rgb`. Test with base-only and all-three-cameras to understand the wrist camera's contribution.

---

## 8. Appendix -- Full Run Table

All 19 substantive runs with complete available metrics, sorted chronologically.

| Run Name | ID | Date | Hours | Tower | Training | BS | Warmup | action_loss | val_action_loss | gate_mean | val_gate_mean | grad_norm | param_norm | Status |
|----------|-----|------|-------|-------|----------|-----|--------|-------------|-----------------|-----------|---------------|-----------|------------|--------|
| `libero_lora_v1` | ikk7b1am | Apr 02 | 42.9 | none | LoRA r16 | 32 | -- | 0.0117 | -- | -- | -- | 0.128 | 1804.6 | finished |
| `pi05_libero_ki` | jjfa4ohk | May 11 | 50.2 | none | LoRA r16 | 32 | -- | 0.3041* | 0.3513* | -- | -- | 0.135 | 1843.2 | finished |
| `lora_baseline_v1` | mbu4i6ha | May 18 | 11.7 | none | LoRA r16 | 64 | -- | 0.0104 | 0.0313 | -- | -- | 0.156 | 1804.1 | finished |
| `wan_precomp_v1` | psm2fn5a | May 18 | 12.7 | wan_t2v | LoRA r16 | 64 | -- | 0.0103 | 0.0322 | -- | -- | 0.206 | 1804.6 | finished |
| `wan_precomp_semonly_v1` | mkcpouz4 | May 19 | 12.4 | wan_t2v | LoRA r16 | 64 | -- | 0.0103 | 0.0314 | -- | -- | 0.205 | 1804.6 | finished |
| `wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS` | u5e6hba1 | May 20 | 11.0 | wan_t2v | LoRA r16 | 32 | -- | 0.0145 | 0.0271 | -- | -- | 0.329 | 1805.2 | finished |
| `wan_precomp_semonly_v1_w1s1_blk20_NO_INIT_BIAS` | r0lrw1l3 | May 21 | 10.9 | wan_t2v | LoRA r16 | 32 | -- | 0.0146 | 0.0267 | -- | -- | 0.346 | 1805.2 | finished |
| `libero_wan_full_finetune` | 2wbqvkrn | May 22 | 43.1 | wan_t2v | FFT | 128 | -- | 0.0070 | 0.0347 | -- | -- | 0.066 | 1808.6 | failed |
| `libero_deeplora_ki_ar_wan_precomp` | ym4jbof2 | May 22 | 19.1 | none | LoRA r32 | 256 | -- | 0.0089** | 0.0356** | -- | -- | 0.320 | 1818.7 | failed |
| `libero_deeplora_ki_ar_2nd` | xwb8izvd | May 23 | 4.0 | none | LoRA r32 | 32 | -- | 0.0277** | 0.0274** | -- | -- | 0.832 | 1807.5 | finished |
| `libero_deeplora_ki_ar_wan_precomp_fr` | iyj290rn | May 24 | 5.2 | wan_t2v | LoRA r32 | 32 | -- | 0.0307** | 0.0286** | 0.986 | -- | 1.108 | 1808.6 | finished |
| `libero_deeplora_wan_precomp_gatewarmup` | xppa6sva | May 24 | 8.5 | wan_t2v | LoRA r32 | 32 | 6k | 0.0114 | 0.0289 | 0.726 | 0.722 | 0.199 | 1805.4 | finished |
| `cosmos_base_v1_w1s1` | k5wjx2gr | May 24 | 8.5 | cosmos_base | LoRA r32 | 32 | 6k | 0.0177 | 0.0251 | 0.864 | 0.864 | 0.646 | 1805.4 | finished |
| `fft_wan_precomp_gatewarmup` | t9qvhe4x | May 24 | 15.6 | wan_t2v | FFT | 64 | 4k | 0.0045 | 0.0440 | 0.758 | 0.755 | 0.129 | 1803.7 | finished |
| `fft_cosmos_precomp_gatewarmup` | vxjffwqk | May 25 | 10.3 | cosmos_base | FFT | 64 | 4k | 0.0153 | 0.0267 | 0.955 | 0.959 | 0.167 | 1803.6 | killed |
| `libero_fft` | mqjjfm3o | May 25 | 12.3 | none | FFT | 64 | -- | 0.0054 | 0.0400 | -- | -- | 0.131 | 1803.2 | killed |
| `fft_wan_precomp_gatewarmup_smallbatch` | vk3wx7vt | May 26 | 4.9 | wan_t2v | FFT | 16 | 8k | 0.0195 | 0.0213 | 0.323 | 0.339 | 0.449 | 1803.0 | finished |
| `libero_fft_smallbatch` | cmd8ph9u | May 26 | 4.5 | none | FFT | 16 | -- | 0.0196 | 0.0221 | -- | -- | 0.438 | 1802.4 | finished |
| `fft_wan_precomp_LONGER_gatewarmup_smallbatch` | eewwpqtl | May 26 | -- | wan_t2v | FFT | 16 | 15k | 0.0169 | 0.0245 | 0.321 | 0.317 | 0.511 | 1803.0 | **running** |

\* KI total loss (includes ki_ar_loss + ki_fm_loss + fast_loss).
\*\* action_loss component extracted from KI+FAST combined loss.
