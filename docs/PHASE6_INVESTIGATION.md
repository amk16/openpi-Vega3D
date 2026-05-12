# Phase 6 Investigation — Ground Truth for DreamDojo Integration

**Date:** 2026-05-12
**Scope:** Sub-Phase 6.0 — answer the open questions before writing any tower code.

---

## 1. Architectural Delta: DreamDojo vs WAN Tower

### What exists (WAN tower, for comparison)

The WAN T2V tower (`src/openpi_vega3d/towers/wan_tower.py`) uses a custom WAN model loaded via `WanModel.from_pretrained()`. Key properties:

| Property | WAN T2V 1.3B | Source |
|----------|-------------|--------|
| Loader | `WanModel.from_pretrained()` (custom loader) | `wan_t2v_encoder.py:87` |
| VAE | `WanVAE` (custom) | `wan_t2v_encoder.py:67-71` |
| VAE spatial stride | `cfg.vae_stride` (config-driven) | `wan_t2v_encoder.py:62` |
| Patch size | `cfg.patch_size` (config-driven) | `wan_t2v_encoder.py:63` |
| Block list attribute | `self.model.blocks` | `wan_t2v_encoder.py:191` |
| Forward hook | `self.model.blocks[block_idx].register_forward_hook(...)` | `wan_t2v_encoder.py:195` |
| feat_dim | 1536 (post-MLP residual-stream width, not `cfg.dim=1280`) | `wan_tower.py:17-23` |
| Prompt embedding | Pre-computed `.pt` file loaded at construction | `wan_t2v_encoder.py:77-85` |
| Noise schedule | `FlowUniPCMultistepScheduler`, `timestep=300`, `shift=5.0` | `wan_t2v_encoder.py:88-92` |
| Spatial pooling | `F.adaptive_avg_pool2d(feats, output_size=(output_spatial, output_spatial))` | `wan_t2v_encoder.py:222` |

### What DreamDojo / Cosmos-Predict2.5 brings

DreamDojo is built on NVIDIA's Cosmos-Predict2.5, a latent video diffusion model. The base Cosmos-Predict2.5 model is available via HuggingFace diffusers; DreamDojo adds action conditioning on top.

| Property | Cosmos-Predict2.5-2B | Source |
|----------|---------------------|--------|
| Loader | `Cosmos2_5_PredictBasePipeline.from_pretrained()` (diffusers) | [HF diffusers docs](https://huggingface.co/docs/diffusers/en/api/pipelines/cosmos) |
| Transformer class | `CosmosTransformer3DModel` | [diffusers source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos.py) |
| VAE | Cosmos CausalContinuousVideoTokenizer (8x spatial compression) | [NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) |
| VAE latent channels | 16 (vs WAN's variable) | `in_channels=16` in transformer config |
| Patch size | `(1, 2, 2)` — temporal=1, spatial=2x2 | CosmosTransformer3DModel defaults |
| **Block list attribute** | **`self.transformer_blocks`** (NOT `blocks`) | diffusers source code |
| Block class | `CosmosTransformerBlock` | diffusers source code |
| num_layers | **28** transformer blocks | CosmosTransformer3DModel defaults |
| Hidden size | **To verify in 6.2** — likely 2048 for 2B (`num_attention_heads * attention_head_dim`) | Needs introspection |
| Total params | ~2.06B | [HF model card](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) |
| Text encoder | Cosmos-Reason1 (VLM) | HF model card |
| Text conditioning | Cross-attention via `encoder_hidden_states` | diffusers forward signature |
| Timestep conditioning | AdaLN (`CosmosAdaLayerNormZero`) — timestep only | diffusers source code |
| **Action conditioning** | **NOT present in base model** | diffusers forward: no action arg |
| Forward signature | `(hidden_states, timestep, encoder_hidden_states, block_controlnet_hidden_states, attention_mask, fps, condition_mask, padding_mask, return_dict)` | diffusers source code |

### Key architectural differences from WAN

1. **Loading API.** WAN uses custom `WanModel.from_pretrained()`. Cosmos uses standard diffusers `Cosmos2_5_PredictBasePipeline.from_pretrained()` or `CosmosTransformer3DModel.from_pretrained()`. Different API surface entirely.

2. **Block attribute path.** WAN: `self.model.blocks`. Cosmos: `self.transformer_blocks`. The forward hook registration must use the correct attribute.

3. **VAE.** WAN: custom `WanVAE` with variable compression. Cosmos: `CausalContinuousVideoTokenizer` with 8x spatial compression per dimension and 16 latent channels.

4. **Text conditioning.** WAN: pre-computed prompt embedding passed as `context` tensor. Cosmos: text encoded by Cosmos-Reason1 and passed as `encoder_hidden_states` for cross-attention. For feature extraction with zeroed conditioning, we zero the `encoder_hidden_states` tensor.

5. **No action input in base model.** This is the single biggest difference. DreamDojo's paper adds "continuous latent actions" via chunked injection into AdaLN, but the publicly released Cosmos-Predict2.5-2B base model's `forward()` has no action argument. For Phase 6 (null action regime), this is actually simpler — we just don't pass actions at all.

---

## 2. Five Concrete Blockers

### Blocker 1: Model loading path

**Where:** New file `src/openpi_vega3d/towers/dreamdojo_tower.py`

WAN loads via a custom `WanModel.from_pretrained()` that reads checkpoint files directly. Cosmos-Predict2.5 loads via the diffusers ecosystem:

```python
from diffusers import Cosmos2_5_PredictBasePipeline
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    "nvidia/Cosmos-Predict2.5-2B",
    revision="diffusers/base/post-trained",
    torch_dtype=torch.bfloat16,
)
transformer = pipe.transformer  # CosmosTransformer3DModel
vae = pipe.vae                  # AutoencoderKLCosmos (likely)
```

**Risk:** Loading the full pipeline pulls in the text encoder (Cosmos-Reason1) which may be large and unnecessary for feature extraction. May want to load just the transformer and VAE components separately using `CosmosTransformer3DModel.from_pretrained()` and `AutoencoderKL.from_pretrained()` with `subfolder=` args.

**Resolution:** Sub-phase 6.2 must test both approaches and pick the lighter one.

### Blocker 2: Forward hook on correct block attribute

**Where:** `wan_t2v_encoder.py:191-195` (reference pattern)

WAN hooks into `self.model.blocks[block_idx]`. Cosmos uses `self.transformer_blocks` (confirmed from diffusers source). The hook pattern is identical in concept but the attribute path differs:

```python
# WAN:
handle = self.model.blocks[block_idx].register_forward_hook(_hook)

# Cosmos:
handle = transformer.transformer_blocks[block_idx].register_forward_hook(_hook)
```

**Resolution:** Sub-phase 6.2 verifies the attribute exists and contains `CosmosTransformerBlock` instances.

### Blocker 3: feat_dim introspection

**Where:** `wan_tower.py:57` (reference pattern)

WAN reads `cfg.dim` from its config. Cosmos exposes config differently:

```python
# WAN:
self._feat_dim = getattr(self.encoder.cfg, "dim", 1280)

# Cosmos (expected):
self._feat_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
```

**Open question:** The diffusers defaults show `num_attention_heads=32, attention_head_dim=128` → hidden_size=4096. But the 2B model likely uses smaller values (hidden_size≈2048 based on the patch embedding projection layer analysis). Sub-phase 6.2 MUST print `transformer.config` and compute the actual hidden_size.

**Why this matters:** `P_gen = nn.Linear(feat_dim, 2048)` in `pi0_pytorch.py:130`. If feat_dim=4096 instead of 2048, P_gen becomes ~8M params instead of ~4M. Still fine, but changes the adapter parameter budget.

### Blocker 4: Spatial math — CRITICAL CORRECTION

**Where:** Plan's locked decision #3 (spatial-grid strategy)

**The plan's math is wrong.** The plan says "Feed 448×448 so WAN2.2's 4×16×16 compression yields a 14×14 latent grid." This is incorrect for Cosmos-Predict2.5.

Cosmos-Predict2.5 spatial stride calculation:
- VAE spatial compression: **8x per dimension** (CausalContinuousVideoTokenizer, `CV8x8x8` variant)
- DiT patch size: **(1, 2, 2)** — 2x spatial downsampling in patchification
- Total effective stride per spatial dimension: **8 × 2 = 16**

Token count at various input resolutions:

| Input resolution | After VAE (÷8) | After patchify (÷2) | Token count | Notes |
|-----------------|----------------|---------------------|-------------|-------|
| 224×224 | 28×28 | 14×14 | 196 | Same as WAN default |
| 256×256 | 32×32 | 16×16 | **256** | **Exact PaliGemma match — no pooling needed** |
| 448×448 | 56×56 | 28×28 | 784 | Plan's suggestion — far too many tokens |

**Corrected decision:** Use **input_resolution=256** (not 448). This produces exactly 16×16 = 256 tokens, matching PaliGemma's native SigLIP grid with NO pooling or interpolation. This is cleaner than any alternative.

If the VAE compression turns out to be different (verified in 6.2), the formula is:
```
input_resolution = desired_grid_size × VAE_spatial_stride × patch_spatial_size
                 = 16 × VAE_stride × 2
```

### Blocker 5: Text embedding for zeroed conditioning

**Where:** `wan_t2v_encoder.py:77-85` (reference pattern)

WAN loads a pre-computed prompt embedding and passes it as `context`. Cosmos uses `encoder_hidden_states` from a text encoder (Cosmos-Reason1). For null-text conditioning, we need to:

1. Determine the expected shape of `encoder_hidden_states` from the transformer config (`text_embed_dim`, sequence length)
2. Pass a zero tensor of that shape

The `text_embed_dim` default in diffusers is 4096 (large model), but the 2B variant may use 1024. Must verify in 6.2.

**Resolution:** In 6.2, print `transformer.config.text_embed_dim` and create the zero text tensor accordingly.

---

## 3. Locked Decisions

### Decision 1: Use DreamDojo 2B pretrain checkpoint

**Value:** Load DreamDojo's 2B pretrained weights from `nvidia/DreamDojo` on HuggingFace (`2B_pretrain/iter_000140000/model/`).

**Why:**
- DreamDojo is the whole point — 44k hours of egocentric human video gives robotics-relevant geometric features that base Cosmos lacks
- DreamDojo IS Cosmos-Predict2.5 architecturally (same transformer), with added action conditioning layers
- Loading strategy: instantiate `CosmosTransformer3DModel` via diffusers for the architecture, then load DreamDojo `.pt` weights with `strict=False` — extra action-conditioning keys are skipped (same pattern as `policy_utils.py:100-107` for VEGA-3D adapter keys)
- For Phase 6 (null-action feature extraction), we only need the base transformer weights (self-attention, cross-attention, feedforward, norms). The action-conditioning layers are unused.

**Checkpoint format:** DreamDojo weights are in DCP (Distributed CheckPoint) format. One-time conversion via `convert_distcp_to_pt.py` from the cosmos-predict2.5 repo produces a `.pt` file loadable by PyTorch.

**Fallback:** If `strict=False` loading fails (architecture mismatch beyond extra keys), use the cosmos-predict2.5 codebase's native loader instead of diffusers.

**Variant:** Teacher only (the pretrained checkpoint). DreamDojo's distilled student runs at 10 FPS but does not support intermediate-noise feature extraction. Refuse `variant="student"` at construction time.

### Decision 2: Action regime = null (simplified)

**Value:** No action input. The base Cosmos-Predict2.5-2B model's forward signature has no action argument.

**Why:**
- The plan's original "Regime A" (zero 32-d × 4-stacked action tensor at AdaLN slot) was designed for DreamDojo's specialized model which has action conditioning
- Since we're using the base Cosmos model (Decision 1), there's no action slot to zero out
- The feature extraction is purely geometric: VAE encode → add noise → denoise one step → hook intermediate features
- This is identical in spirit to how the WAN tower works (WAN also has no action conditioning)

**Future extension:** Phase 7 can evaluate switching to DreamDojo's specialized checkpoint with actual action conditioning. At that point, the null regime becomes "zero the action input" and averaged/swept regimes become meaningful.

### Decision 3: Spatial-grid strategy = input_resolution 256 (CORRECTED from plan)

**Value:** Feed 256×256 images to the tower. With Cosmos's 8x VAE + 2x2 patchify = 16x total stride, this produces exactly 16×16 = 256 tokens.

**Why:**
- 256 tokens matches PaliGemma's native SigLIP grid exactly
- **No pooling or interpolation needed** — cleaner than the plan's 448→14×14→pool-to-16×16 proposal
- The plan's 448 suggestion was based on an incorrect compression ratio assumption. Cosmos uses 8x spatial VAE (not 16x), and the DiT patchifies 2x2 on top. Total stride is 16, not 32.
- 256×256 is a standard resolution that both the VAE and patchifier divide cleanly

**Contingency:** If sub-phase 6.2 reveals a different VAE compression ratio, recalculate:
```
input_resolution = 16 × VAE_spatial_stride × patch_spatial_size
```

### Decision 4: Layer-depth default = round(0.7 × num_blocks)

**Value:** `feat_block_idx = round(0.7 × 28) = 20` for the 2B model (28 blocks).

**Why:**
- VEGA-3D paper recommends ~70% fractional depth for feature extraction
- The existing WAN tower defaults to `feat_block_idx=-1` (last block), which is a discrepancy with the paper
- Using 70% depth from the start avoids carrying forward the WAN tower's deviation
- Blocks 0-27 are available; block 20 is at 71.4% depth

**Note:** The WAN tower's `-1` default (last block) should be corrected separately, but this is outside Phase 6 scope.

### Decision 5: Camera selection — both base and wrist configs

**Value:** Two TrainConfig entries: `pi05_b1k_dreamdojo` (base camera) and `pi05_b1k_dreamdojo_wrist` (wrist cameras).

**Why:** Same reasoning as the plan. DreamDojo's egocentric training distribution may be a better viewpoint match for wrist cameras. Having both configs enables direct comparison in Phase 7.

---

## 4. Dependency Graph for Sub-Phases 6.1–6.8

```
6.0  Investigation (this doc)          ← YOU ARE HERE
 │
 ▼
6.1  Skeleton scaffold
 │   Create DreamDojoTower(BaseTower) placeholder returning dummy tensors.
 │   Register "dreamdojo" in TOWER_REGISTRY.
 │
 ▼
6.2  Real loader + feat_dim introspection
 │   Load Cosmos-Predict2.5-2B via diffusers.
 │   Verify: hidden_size, block attribute path, VAE compression ratio.
 │   Resolve: Blocker 1 (loader), Blocker 3 (feat_dim), Blocker 4 (spatial math).
 │
 ▼
6.3  Null-text forward pass
 │   Real encode() with zero text embeddings.
 │   Hook intermediate DiT block features.
 │   Resolve: Blocker 2 (hook path), Blocker 5 (text embedding shape).
 │
 ▼
6.4  Spatial-grid output
 │   Feed 256×256 input → 16×16 = 256 tokens (if VAE is 8x).
 │   Verify output shape matches BaseTower contract.
 │
 ├────────────────────────┐
 ▼                        ▼
6.5  TrainConfig          6.7  test_tower.py validation
 │   pi05_b1k_dreamdojo        Run existing offline + online tests.
 │   entry.
 │
 ▼
6.6  Camera-choice config
 │   pi05_b1k_dreamdojo_wrist
 │
 └────────────────────────┐
                          ▼
                         6.8  Documentation + cleanup
```

6.5/6.6 and 6.7 are independent after 6.4. All must complete before 6.8.

---

## 5. Open Questions and Risks

### Must-resolve in 6.2 (before any real code)

| # | Question | How to resolve | Impact if different from expected |
|---|----------|---------------|-----------------------------------|
| 1 | What is `transformer.config.num_attention_heads × attention_head_dim` for the 2B model? | `print(pipe.transformer.config)` | Changes `P_gen` input dim and adapter param count |
| 2 | What is the actual VAE spatial compression ratio? | Encode a known-size image, measure latent dims | Changes `input_resolution` calculation |
| 3 | Is the block list at `transformer.transformer_blocks`? | `print(type(pipe.transformer.transformer_blocks))` | Changes hook registration code |
| 4 | What is `text_embed_dim` for the 2B model? | Read from `transformer.config` | Changes zero text tensor shape |
| 5 | Does the pipeline load Cosmos-Reason1 text encoder automatically? How large is it? | Try loading, measure memory | May need to load transformer+VAE only to save memory |

### Known risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| **Cosmos-Predict2.5 diffusers integration is immature or broken** | Low (released Dec 2025, 5+ months ago) | Fallback: load via nvidia-cosmos/cosmos-predict2.5 GitHub package's own loader |
| **Pipeline loads full text encoder, wasting memory** | Medium | Load transformer and VAE separately via `CosmosTransformer3DModel.from_pretrained()` |
| **feat_dim is 4096 (not 2048), making P_gen too large** | Medium | P_gen would be ~8M params. Still within budget (10M threshold). Adjust param count assertion. |
| **VAE compression is not 8x** | Low (Cosmos tokenizer naming convention `CV8x8x8` is clear) | Recalculate input_resolution formula |
| **Forward hook output shape differs from WAN's** | Medium | Hook captures block output as `[B, seq_len, hidden_size]`. May need different reshape logic than WAN. |
| **Dual-tower memory for wrist config** | Medium | Two tower forwards at 256×256 + Pi0 backbone. If tight: share tower instance, process cameras sequentially. |

---

## 6. Corrections to the Original Plan

The investigation surfaced three corrections to `docs/PHASE6_PLAN.md` that should be applied before starting implementation:

### Correction 1: input_resolution 256, not 448

The plan's spatial-grid strategy (locked decision #3) assumed a compression ratio that yields 14×14 at 448. Cosmos-Predict2.5 actually yields 28×28=784 tokens at 448. The correct input for 256 tokens is **256×256**.

**Action:** Update PHASE6_PLAN.md sub-phases 6.4, 6.5, 6.6 to reference `input_resolution=256`.

### Correction 2: Action regime is simpler than planned

The plan described zeroing a "32-d × 4-stacked action tensor." Since we're using base Cosmos-Predict2.5 (not DreamDojo's specialized checkpoint), there's no action input at all. The null regime simply means "standard denoising feature extraction with no action conditioning" — identical in spirit to how the WAN tower works.

**Action:** Sub-phase 6.3 description should say "null-text forward pass" rather than "null-action + null-text forward pass." No action tensor construction needed.

### Correction 3: Block attribute is `transformer_blocks`, not `blocks`

The plan flagged this as uncertain. It's now confirmed: Cosmos uses `self.transformer_blocks` (from diffusers source). WAN uses `self.model.blocks`.

**Action:** Sub-phase 6.3 implementation should use `pipe.transformer.transformer_blocks[block_idx]`.

---

## 7. Comparison: DreamDojo Tower vs WAN Tower Implementation

Side-by-side comparison of how the new tower maps to the existing WAN pattern:

| Aspect | WAN Tower | DreamDojo Tower (planned) |
|--------|-----------|--------------------------|
| **Constructor** | `WanT2VTower(checkpoint_dir, *, prompt_emb_path, task, size, timestep, shift, feat_block_idx, output_spatial, dtype)` | `DreamDojoTower(checkpoint_dir, *, variant, input_resolution, timestep, feat_block_idx, output_spatial, dtype)` |
| **Model loader** | `WanModel.from_pretrained(checkpoint_dir)` | `CosmosTransformer3DModel.from_pretrained(checkpoint_dir, ...)` |
| **VAE loader** | `WanVAE(vae_pth=...)` | Load from pipeline or separate `AutoencoderKL.from_pretrained(...)` |
| **Prompt/text** | Pre-computed `.pt` embedding → `context` tensor | Zero tensor of shape `[1, seq_len, text_embed_dim]` → `encoder_hidden_states` |
| **Block hook** | `self.model.blocks[idx]` | `self.transformer.transformer_blocks[idx]` |
| **feat_dim source** | `cfg.dim` (1280 config, but actual is 1536 at hook) | `config.num_attention_heads * config.attention_head_dim` (verify at hook) |
| **Noise schedule** | `FlowUniPCMultistepScheduler` (custom WAN) | Cosmos pipeline scheduler (flow-matching based) |
| **Spatial pooling** | `adaptive_avg_pool2d(output_spatial)` | None needed if input_resolution=256 → 16×16=256 tokens. Otherwise same pooling. |
| **encode() output** | `[B, output_spatial², feat_dim]` | `[B, 256, feat_dim]` |

---

## 8. Ready State

All five investigation questions have answers (some pending 6.2 empirical verification). No hidden blockers beyond what's documented. Critical spatial-math correction identified and documented.

Next step: **Sub-Phase 6.1 — Skeleton scaffold** with placeholder `DreamDojoTower(BaseTower)` returning dummy tensors and registered in TOWER_REGISTRY.
