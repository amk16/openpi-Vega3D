#!/usr/bin/env python
"""Inspect the VEGA-3D adaptive-fusion gate on LIBERO *validation* frames.

What this does
--------------
1. Syncs a ``pi05_libero_lora_wan_precomp`` checkpoint from S3 (``params/`` +
   ``assets/`` only -- the 5GB optimizer ``train_state/`` is skipped).
2. Builds a data loader over the held-out LIBERO validation episodes -- the
   exact same split scripts/train.py uses (``config.val_episodes_index``,
   sampled with ``shuffle_seed=config.seed``), so the frames are guaranteed to
   come from the validation portion, not the training portion.
3. Loads the JAX Pi0 model from the checkpoint.
4. Runs inference, recording the fusion gate ``g`` produced by every
   ``AdaptiveGatedFusion`` call, and drops into ``breakpoint()`` so you can
   inspect it.

Reading the gate (see src/openpi/models/adaptive_gated_fusion.py)
-----------------------------------------------------------------
    F_fused = (1 - g) * F_gen + g * F_sem

  * ``g``      -- weight on F_sem, the semantic / SigLIP stream.
  * ``1 - g``  -- weight on F_gen, i.e. the *new* P_gen / WAN generative stream.

The gate is biased toward semantic at init (gate_init_bias=4.0):
    g0 = sigmoid(4.0) ~= 0.9820   ->   P_gen weight (1 - g0) ~= 0.0180

So "how much the gate opened up in favor of P_gen" == how far the mean of
``1 - g`` has risen above ~0.018.

Usage
-----
    python scripts/inspect_gate_weights.py                 # full inference
    python scripts/inspect_gate_weights.py --prefix-only   # faster: skip the
                                                           # action-sampling loop
    python scripts/inspect_gate_weights.py --num-frames 16
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import subprocess

import numpy as np

DEFAULT_S3 = (
    "s3://behavior-challenge/openpi_checkpoints/"
    "pi05_libero_lora_wan_precomp/wan_precomp_v1/8000/"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect VEGA-3D gate weights on LIBERO validation frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config-name", default="pi05_libero_lora_wan_precomp",
                   help="TrainConfig name to load the model/data pipeline from.")
    p.add_argument("--s3-checkpoint", default=DEFAULT_S3,
                   help="S3 URI of the checkpoint step directory.")
    p.add_argument("--local-ckpt-dir", default=None,
                   help="Local directory to sync the checkpoint into "
                        "(default: <repo>/checkpoints/<config>/wan_precomp_v1/8000).")
    p.add_argument("--num-frames", type=int, default=8,
                   help="Number of validation frames (batch size) to run.")
    p.add_argument("--num-steps", type=int, default=10,
                   help="Flow-matching denoising steps for sample_actions.")
    p.add_argument("--prefix-only", action="store_true",
                   help="Only run embed_prefix (SigLIP + fusion). The gate is "
                        "computed there, so this captures it without paying for "
                        "the full action-sampling LLM loop.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument("--skip-download", action="store_true",
                   help="Assume the checkpoint is already present locally.")
    return p.parse_args()


def sync_checkpoint(s3_uri: str, local_dir: pathlib.Path, *, skip: bool) -> pathlib.Path:
    """Sync params/ and assets/ from the S3 checkpoint dir. Skips train_state/."""
    local_dir.mkdir(parents=True, exist_ok=True)
    params_dir = local_dir / "params"
    if params_dir.exists() and any(params_dir.iterdir()):
        print(f"[ckpt] reusing cached checkpoint at {local_dir}")
        return local_dir
    if skip:
        raise SystemExit(f"[ckpt] --skip-download set but no params/ under {local_dir}")
    base = s3_uri.rstrip("/")
    for sub in ("params", "assets"):
        src, dst = f"{base}/{sub}", local_dir / sub
        print(f"[ckpt] aws s3 sync {src} -> {dst}")
        subprocess.run(["aws", "s3", "sync", src, str(dst)], check=True)
    return local_dir


def main() -> None:
    args = parse_args()

    # Heavy imports kept after argparse so --help stays instant.
    import jax
    import jax.numpy as jnp

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    from openpi.models import adaptive_gated_fusion as _agf

    local_ckpt_dir = pathlib.Path(
        args.local_ckpt_dir
        or REPO_ROOT / "checkpoints" / args.config_name / "wan_precomp_v1" / "8000"
    )
    local_ckpt = sync_checkpoint(args.s3_checkpoint, local_ckpt_dir, skip=args.skip_download)

    # ------------------------------------------------------------------
    # Config + validation data loader
    # ------------------------------------------------------------------
    config = _config.get_config(args.config_name)
    if not config.val_episodes_index:
        raise SystemExit(
            f"{args.config_name} has no val_episodes_index -- cannot isolate a "
            "validation split."
        )
    # One batch in the main process: no need for the 20-worker training pool.
    config = dataclasses.replace(config, num_workers=0)
    val_episodes = list(config.val_episodes_index)
    print(f"[data] LIBERO validation split: {len(val_episodes)} held-out episodes "
          f"(episode indices {val_episodes[:5]}... every 20th)")

    # Mirrors the validation loader in scripts/train.py: the held-out episodes
    # only, wrapped in a fixed seeded permutation so the sampled frames are a
    # representative draw across all validation episodes.
    val_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        episodes_index=val_episodes,
        batch_size=args.num_frames,
        num_batches=1,
        shuffle_seed=config.seed,
    )
    obs, actions = next(iter(val_loader))
    print(f"[data] loaded {args.num_frames} validation frames; "
          f"images={ {k: tuple(v.shape) for k, v in obs.images.items()} }")
    if obs.tower_features is None:
        raise SystemExit(
            "observation.tower_features is None -- the precomputed WAN features "
            "were not loaded. Check tower_features_cache_dir in the config."
        )
    print(f"[data] tower_features cameras: "
          f"{ {k: tuple(v.shape) for k, v in obs.tower_features.items()} }")

    # ------------------------------------------------------------------
    # Load the trained model
    # ------------------------------------------------------------------
    print(f"[model] restoring params from {local_ckpt / 'params'} ...")
    params = _model.restore_params(local_ckpt / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    if not getattr(model, "use_vega3d", False) or model.fusion is None:
        raise SystemExit("Loaded model has no VEGA-3D fusion module.")
    print("[model] loaded; VEGA-3D adaptive gated fusion is active.")

    # ------------------------------------------------------------------
    # Instrument AdaptiveGatedFusion so every gate computation is recorded.
    # We recompute g exactly as adaptive_gated_fusion.py does (a read-only
    # peek), then defer to the original __call__ so the model output is
    # byte-identical to an un-instrumented run.
    # ------------------------------------------------------------------
    gate_records: list[np.ndarray] = []
    orig_call = _agf.AdaptiveGatedFusion.__call__

    def recording_call(self, f_gen, f_sem):
        if self.force_gate is not None:
            g = jnp.full((*f_gen.shape[:-1], 1), self.force_gate, dtype=f_gen.dtype)
        else:
            concat = jnp.concatenate([self.ln_gen(f_gen), self.ln_sem(f_sem)], axis=-1)
            g = jax.nn.sigmoid(self.gate_proj(concat))
        gate_records.append(np.asarray(jax.device_get(g), dtype=np.float32))
        return orig_call(self, f_gen, f_sem)

    _agf.AdaptiveGatedFusion.__call__ = recording_call
    actions_pred = None
    try:
        if args.prefix_only:
            print("[infer] running embed_prefix only (SigLIP + fusion) ...")
            pre = _model.preprocess_observation(None, obs, train=False)
            model.embed_prefix(pre)
        else:
            print(f"[infer] running sample_actions (num_steps={args.num_steps}); "
                  "this is un-jitted so the gate stays a concrete array ...")
            actions_pred = np.asarray(jax.device_get(
                model.sample_actions(jax.random.key(args.seed), obs,
                                     num_steps=args.num_steps)
            ))
            print(f"[infer] predicted actions shape: {actions_pred.shape}")
    finally:
        _agf.AdaptiveGatedFusion.__call__ = orig_call

    # ------------------------------------------------------------------
    # Organize the recorded gates. embed_prefix runs the fusion once per
    # spatial camera, in IMAGE_KEYS order, filtered to vega3d_cameras.
    # ------------------------------------------------------------------
    spatial = tuple(config.model.vega3d_cameras)
    camera_order = [k for k in _model.IMAGE_KEYS if k in spatial]
    if len(gate_records) != len(camera_order):
        print(f"[warn] recorded {len(gate_records)} gate calls but expected "
              f"{len(camera_order)} (cameras {camera_order}); labeling by index.")

    G_INIT = 1.0 / (1.0 + math.exp(-4.0))  # gate at init: sigmoid(gate_init_bias)
    gates: list[dict] = []
    for i, g in enumerate(gate_records):
        cam = camera_order[i] if i < len(camera_order) else f"call_{i}"
        flat = g.reshape(g.shape[0], -1)            # [B, N]  (N = 256 tokens)
        n = flat.shape[1]
        side = int(round(math.sqrt(n)))
        gates.append({
            "camera": cam,
            "g": flat,                              # weight on F_sem (semantic)
            "p_gen_weight": 1.0 - flat,             # weight on F_gen (new P_gen/WAN)
            "g_grid": g.reshape(g.shape[0], side, side) if side * side == n else None,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    gate_bias = np.asarray(jax.device_get(model.fusion.gate_proj.bias.value), np.float32)
    gate_kernel = np.asarray(jax.device_get(model.fusion.gate_proj.kernel.value), np.float32)

    print("\n" + "=" * 74)
    print("VEGA-3D ADAPTIVE FUSION GATE  --  F_fused = (1 - g) * F_gen + g * F_sem")
    print("=" * 74)
    print(f"  init gate     : g0 = sigmoid(4.0) = {G_INIT:.4f}  "
          f"=>  P_gen weight (1 - g0) = {1 - G_INIT:.4f}")
    print(f"  learned bias  : gate_proj.bias = {gate_bias.ravel()[0]:+.4f}  "
          f"(init was +4.0000)")
    print(f"  kernel ||W||  : {np.linalg.norm(gate_kernel):.4f}  "
          "(zero at init; non-zero => gate is input-dependent)")
    all_pgen = []
    for gd in gates:
        g, pg = gd["g"], gd["p_gen_weight"]
        all_pgen.append(pg.ravel())
        print(f"\n  camera: {gd['camera']}   (g shape per frame = {g.shape[1:]} tokens)")
        print(f"    g       (semantic weight): "
              f"mean={g.mean():.4f}  std={g.std():.4f}  "
              f"min={g.min():.4f}  max={g.max():.4f}")
        print(f"    1 - g   (P_gen   weight) : "
              f"mean={pg.mean():.4f}  std={pg.std():.4f}  "
              f"min={pg.min():.4f}  max={pg.max():.4f}")
        for q in (10, 50, 90, 99):
            print(f"      P_gen weight  p{q:<2d} = {np.percentile(pg, q):.4f}")
    overall = np.concatenate(all_pgen)
    print("\n  " + "-" * 70)
    print(f"  OVERALL mean P_gen weight (1 - g) = {overall.mean():.4f}")
    print(f"  vs init {1 - G_INIT:.4f}  =>  the gate opened {overall.mean() / (1 - G_INIT):.1f}x "
          "toward the new P_gen stream.")
    print("=" * 74)

    # Convenience handles for the breakpoint.
    g_by_camera = {gd["camera"]: gd["g"] for gd in gates}
    pgen_by_camera = {gd["camera"]: gd["p_gen_weight"] for gd in gates}

    print(
        "\nDropping into breakpoint(). Inspect:\n"
        "  gates           - list of per-camera dicts: 'camera', 'g',\n"
        "                    'p_gen_weight' (=1-g), 'g_grid' ([B,16,16] spatial map)\n"
        "  g_by_camera     - {camera: g array [B, N]}        (semantic weight)\n"
        "  pgen_by_camera  - {camera: 1-g array [B, N]}      (new P_gen weight)\n"
        "  gate_records    - raw [B, N, 1] gate arrays in call order\n"
        "  gate_bias       - learned gate_proj bias (init was +4.0)\n"
        "  gate_kernel     - learned gate_proj kernel [2*D_llm, 1]\n"
        "  G_INIT          - gate value at init (~0.982)\n"
        "  obs, actions    - the validation batch (Observation, ground-truth actions)\n"
        "  actions_pred    - predicted actions (None if --prefix-only)\n"
        "  model           - the loaded Pi0 model (model.fusion is the gate module)\n"
    )
    breakpoint()


if __name__ == "__main__":
    main()
