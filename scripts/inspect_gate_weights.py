#!/usr/bin/env python
"""Print the mean VEGA-3D gate value on LIBERO validation frames.

Usage:
    python scripts/inspect_gate_weights.py
    python scripts/inspect_gate_weights.py --s3-checkpoint s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora_wan_precomp/wan_precomp_v1_w1s1_blk20/8000/
    python scripts/inspect_gate_weights.py --skip-download --local-ckpt-dir /path/to/ckpt
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess

DEFAULT_S3 = (
    "s3://behavior-challenge/openpi_checkpoints/"
    "pi05_libero_lora_wan_precomp/wan_precomp_v1/8000/"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero_lora_wan_precomp")
    p.add_argument("--s3-checkpoint", default=DEFAULT_S3)
    p.add_argument("--local-ckpt-dir", default=None)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def sync_checkpoint(s3_uri: str, local_dir: pathlib.Path, *, skip: bool) -> pathlib.Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    if skip:
        if not (local_dir / "params").exists():
            raise SystemExit(f"--skip-download but no params/ under {local_dir}")
        return local_dir
    base = s3_uri.rstrip("/")
    for sub in ("params", "assets"):
        src, dst = f"{base}/{sub}", local_dir / sub
        print(f"[ckpt] aws s3 sync {src} -> {dst}")
        subprocess.run(["aws", "s3", "sync", src, str(dst)], check=True)
    return local_dir


def main() -> None:
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    # Derive local checkpoint path from S3 URI if not specified
    if args.local_ckpt_dir:
        local_ckpt_dir = pathlib.Path(args.local_ckpt_dir)
    else:
        parts = args.s3_checkpoint.rstrip("/").split("/")
        local_ckpt_dir = REPO_ROOT / "checkpoints" / parts[-3] / parts[-2] / parts[-1]

    local_ckpt = sync_checkpoint(args.s3_checkpoint, local_ckpt_dir, skip=args.skip_download)

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, num_workers=0)
    val_episodes = list(config.val_episodes_index)

    val_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        episodes_index=val_episodes,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        shuffle_seed=config.seed,
    )

    print(f"[model] restoring params from {local_ckpt / 'params'} ...")
    params = _model.restore_params(local_ckpt / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print(f"[model] loaded; use_vega3d={model.use_vega3d}")

    gate_means = []
    for i, (obs, actions) in enumerate(val_loader):
        obs = _model.preprocess_observation(None, obs, train=False)
        _, _, _, avg_gate = model.embed_prefix(obs)
        if avg_gate is not None:
            gate_means.append(float(jax.device_get(avg_gate)))
            print(f"  batch {i}: gate_mean = {gate_means[-1]:.6f}")
        else:
            print(f"  batch {i}: no gate (VEGA-3D not active)")
            break

    if gate_means:
        overall = np.mean(gate_means)
        print(f"\n  OVERALL gate_mean = {overall:.4f}")
        print(f"  => {overall*100:.1f}% SigLIP, {(1-overall)*100:.1f}% Wan")


if __name__ == "__main__":
    main()
