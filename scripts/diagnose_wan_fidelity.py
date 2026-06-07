#!/usr/bin/env python
"""Phase 8.0: empirical diagnostics for the WAN fusion fidelity breaks.

Two subcommands, each targeting one break mechanism from the 2026-06 fidelity
audit (research-wiki: wan-fusion-fidelity-breaks). Run BOTH against the
existing (pre-fix) artifacts before regenerating anything — they are the
"before" half of Phase 9's before/after comparison.

Break 1 (token-grid misregistration, letterbox pad tokens pooled into the
16x16 grid):

    python scripts/diagnose_wan_fidelity.py column-energy \
        --cache_dir tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20 \
        --episodes 5

    Prediction if the break is live in the trained cache: conspicuously
    distinct energy in the ~3 outermost columns each side (pure black-bar
    tokens; pool-bin math for the 52->16 adaptive pool says exactly 3
    pure-pad + 1 mixed output columns per side). Doubles as end-to-end
    cache-provenance validation.

Break 2 (raw-stream blend; scale mismatch between f_gen and f_sem):

    python scripts/diagnose_wan_fidelity.py norm-ratio \
        --config pi05_libero_fft_wan_precomp_gatewarmup --checkpoint <ckpt_dir>

    Logs ||f_gen||_2 / ||f_sem||_2 per token batch (the exact operands of the
    blend in pi0.py::_fuse_camera). A ratio far from 1 (x3 or more either
    way) confirms the scale-mismatch mechanism is live. Omit --checkpoint to
    measure at init.

`column-energy` needs only torch + safetensors (CPU fine, no checkpoint —
runnable anywhere, including CI against a synthetic cache). `norm-ratio`
needs the full JAX training environment + the precomputed feature cache
(remote).

Deliberately out of scope (Break 3 is deferred from Phase 8): a
`noise-variance` subcommand sizing the seeded-vs-fresh noise mismatch —
~30 lines on this same scaffolding if Phase 9's result motivates it.
"""

# ruff: noqa: PLC0415 — imports are deliberately deferred per-subcommand so
# column-energy needs only torch+safetensors (CI / laptop) while norm-ratio
# pulls the full JAX training stack (remote only).

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    n = 0 if max_value <= 0 else round(width * value / max_value)
    return "#" * n


# ---------------------------------------------------------------------------
# column-energy (Break 1)
# ---------------------------------------------------------------------------


def run_column_energy(args: argparse.Namespace) -> None:
    import numpy as np
    import safetensors.torch
    import torch

    cache_dir = pathlib.Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"cache_dir does not exist: {cache_dir}")

    # Provenance: meta.json is the cache's single source of truth.
    meta_path = cache_dir / "meta.json"
    grid = args.grid
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(
            f"[meta] tower={meta.get('tower_name')} window={meta.get('window')} "
            f"stride={meta.get('stride')} blk={meta.get('feat_block_idx')} "
            f"output_spatial={meta.get('output_spatial')} "
            f"content_region_pool={meta.get('tower_kwargs', {}).get('content_region_pool', False)}"
        )
        grid = int(meta.get("output_spatial", grid))
    else:
        print(f"[meta] no meta.json under {cache_dir}; assuming grid={grid}")

    # Camera subdirs: explicit --camera, else every dir containing ep files.
    if args.camera:
        cam_dirs = [cache_dir / args.camera]
    else:
        cam_dirs = sorted(d for d in cache_dir.iterdir() if d.is_dir() and any(d.glob("ep_*.safetensors")))
    if not cam_dirs:
        raise SystemExit(f"No camera dirs with ep_*.safetensors under {cache_dir}")

    for cam_dir in cam_dirs:
        ep_files = sorted(cam_dir.glob("ep_*.safetensors"))[: args.episodes]
        if not ep_files:
            print(f"[{cam_dir.name}] no episode files, skipping")
            continue

        col_energy = torch.zeros(grid, dtype=torch.float64)
        row_energy = torch.zeros(grid, dtype=torch.float64)
        n_frames_total = 0
        for f in ep_files:
            feats = safetensors.torch.load_file(str(f))["features"]  # [T, S*S, C], bf16
            t, n_tokens, c = feats.shape
            if n_tokens != grid * grid:
                raise SystemExit(
                    f"{f.name}: {n_tokens} tokens but grid={grid} implies {grid * grid}. "
                    f"Pass --grid or check meta.json."
                )
            g = feats.to(torch.float32).reshape(t, grid, grid, c)  # row-major [T, H, W, C]
            tok_norm = g.norm(dim=-1)  # [T, H, W] per-token L2
            col_energy += tok_norm.mean(dim=(0, 1)).to(torch.float64)  # mean over frames+rows
            row_energy += tok_norm.mean(dim=(0, 2)).to(torch.float64)  # mean over frames+cols
            n_frames_total += t
        col_energy /= len(ep_files)
        row_energy /= len(ep_files)

        print(f"\n[{cam_dir.name}] {len(ep_files)} episodes, {n_frames_total} frames, grid {grid}x{grid}")
        print("per-COLUMN mean token L2 (Break-1 signature axis):")
        cmax = float(col_energy.max())
        for j in range(grid):
            v = float(col_energy[j])
            print(f"  col {j:2d}  {v:10.3f}  {_ascii_bar(v, cmax)}")
        print("per-ROW mean token L2 (control axis — should be roughly flat):")
        rmax = float(row_energy.max())
        for i in range(grid):
            v = float(row_energy[i])
            print(f"  row {i:2d}  {v:10.3f}  {_ascii_bar(v, rmax)}")

        # Signature check: pool-bin math for 52->16 over an 11-token pillarbox
        # each side predicts output columns {0,1,2} and {13,14,15} are pure pad
        # and {3, 12} are mixed. Compare outer-3 vs middle columns.
        edge = list(range(3)) + list(range(grid - 3, grid))
        middle = list(range(5, grid - 5))
        edge_mean = float(np.mean([col_energy[j] for j in edge]))
        mid_mean = float(np.mean([col_energy[j] for j in middle]))
        ratio = edge_mean / mid_mean if mid_mean > 0 else float("inf")
        print(f"\n  outer-3-columns mean : {edge_mean:.3f}")
        print(f"  middle-columns mean  : {mid_mean:.3f}")
        print(f"  edge/middle ratio    : {ratio:.3f}")
        print("  reading: a ratio conspicuously != 1 with visibly distinct outer columns ON THE")
        print("  COLUMN AXIS ONLY (rows flat) is the Break-1 pillarbox signature. A ratio ~1 on")
        print("  both axes means no pad tokens reached the pooled grid (fix active or break absent).")


# ---------------------------------------------------------------------------
# norm-ratio (Break 2)
# ---------------------------------------------------------------------------


def run_norm_ratio(args: argparse.Namespace) -> None:
    # Heavy imports deferred so column-energy stays torch-light.
    import dataclasses

    import jax
    import jax.numpy as jnp
    import numpy as np

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    config = _config.get_config(args.config)
    config = dataclasses.replace(config, num_workers=0)

    if args.checkpoint:
        ckpt = pathlib.Path(args.checkpoint)
        print(f"[model] restoring params from {ckpt / 'params'} ...")
        params = _model.restore_params(ckpt / "params", dtype=jnp.bfloat16)
        model = config.model.load(params)
    else:
        print("[model] no --checkpoint: measuring at init (random P_gen)")
        model = config.model.create(jax.random.key(args.seed))
    model.eval()
    if not getattr(model, "use_vega3d", False):
        raise SystemExit(f"Config {args.config!r} has use_vega3d=False; nothing to measure.")

    val_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        episodes_index=list(config.val_episodes_index),
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        shuffle_seed=config.seed,
    )

    # Replicates pi0.py::_fuse_camera's blend operands exactly:
    #   f_gen = P_gen(gen_feats); f_sem = P_sem(sem) if P_sem else sem
    # The ratio of these norms is what the raw (legacy) blend mixes.
    ratios: list[float] = []
    for i, (raw_obs, _actions) in enumerate(val_loader):
        obs = _model.preprocess_observation(None, raw_obs, train=False)
        for name in obs.images:
            if name not in model._spatial_cameras:  # noqa: SLF001 — diagnostic introspection
                continue
            if obs.tower_features is None or name not in obs.tower_features:
                print(f"  batch {i} cam {name}: no precomputed tower features in batch, skipping")
                continue
            image_tokens, _ = model.PaliGemma.img(obs.images[name], train=False)
            gen_feats = obs.tower_features[name].astype(image_tokens.dtype)
            f_gen = model.P_gen(gen_feats)
            f_sem = model.P_sem(image_tokens) if model.P_sem is not None else image_tokens
            gen_norm = float(jax.device_get(jnp.linalg.norm(f_gen, axis=-1).mean()))
            sem_norm = float(jax.device_get(jnp.linalg.norm(f_sem, axis=-1).mean()))
            r = gen_norm / sem_norm if sem_norm > 0 else float("inf")
            ratios.append(r)
            print(f"  batch {i} cam {name}: ||f_gen||={gen_norm:10.3f}  ||f_sem||={sem_norm:10.3f}  ratio={r:.3f}")

    if not ratios:
        raise SystemExit("No batches yielded both streams — check cache_dir wiring in the config.")
    overall = float(np.mean(ratios))
    print(f"\n  OVERALL mean ratio ||f_gen||/||f_sem|| = {overall:.3f}")
    print("  reading: a ratio x3+ from 1 (either way) confirms the Break-2 scale-mismatch")
    print("  mechanism — the raw blend's 'g=0.5' is then effectively dominated by one stream.")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("column-energy", help="Break-1 diagnostic: per-column feature energy in a cache")
    p_col.add_argument("--cache_dir", required=True, help="Cache root (contains meta.json + per-camera dirs)")
    p_col.add_argument("--camera", default=None, help="Single camera subdir (default: all found)")
    p_col.add_argument("--episodes", type=int, default=5, help="Number of episode files per camera")
    p_col.add_argument("--grid", type=int, default=16, help="Token grid side (overridden by meta.json)")
    p_col.set_defaults(func=run_column_energy)

    p_nr = sub.add_parser("norm-ratio", help="Break-2 diagnostic: ||f_gen||/||f_sem|| at the blend")
    p_nr.add_argument("--config", default="pi05_libero_fft_wan_precomp_gatewarmup")
    p_nr.add_argument("--checkpoint", default=None, help="Checkpoint dir (with params/); omit to measure at init")
    p_nr.add_argument("--num-batches", type=int, default=5)
    p_nr.add_argument("--batch-size", type=int, default=32)
    p_nr.add_argument("--seed", type=int, default=0, help="Init seed when no checkpoint is given")
    p_nr.set_defaults(func=run_norm_ratio)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
