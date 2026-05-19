"""Probe Wan2.1-T2V-1.3B output shape so vega3d_tower_feat_dim can be set correctly.

Downloads the checkpoint via huggingface_hub if missing, builds the WAN tower at
output_spatial=16, runs a single 224x224 image through, and prints the actual
output shape. Use the printed feat_dim in your TrainConfig
(`vega3d_tower_feat_dim`).

Usage:
    python scripts/probe_wan.py
    python scripts/probe_wan.py --checkpoint_dir /path/to/Wan2.1-T2V-1.3B
"""

import argparse
import os
import pathlib
import subprocess
import sys

import torch

DEFAULT_CKPT_DIR = "/workspace/openpi-Vega3D/ckpts/Wan2.1-T2V-1.3B"
HF_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
PROMPT_EMB_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "openpi_vega3d" / "towers" / "wan_prompt_embedding.pt"
)


def ensure_checkpoint(checkpoint_dir: str) -> None:
    sentinel = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(sentinel):
        print(f"[probe] Checkpoint already present at {checkpoint_dir}")
        return
    from huggingface_hub import snapshot_download
    print(f"[probe] Downloading {HF_REPO} to {checkpoint_dir} (multi-GB, takes a while) ...")
    snapshot_download(HF_REPO, local_dir=checkpoint_dir, local_dir_use_symlinks=False)
    print("[probe] Download complete.")


def ensure_prompt_embedding() -> None:
    if PROMPT_EMB_PATH.exists():
        print(f"[probe] Prompt embedding present at {PROMPT_EMB_PATH}")
        return
    print(f"[probe] Prompt embedding missing at {PROMPT_EMB_PATH}; running export script ...")
    export_script = pathlib.Path(__file__).resolve().parent / "export_wan_prompt_embedding.py"
    subprocess.run([sys.executable, str(export_script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--output_spatial", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_checkpoint(args.checkpoint_dir)
    ensure_prompt_embedding()

    # Import after potential download so the error path of a missing checkpoint
    # surfaces before pulling in the heavy tower deps.
    from openpi_vega3d.towers.wan_tower import WanT2VTower

    print(f"[probe] Building WanT2VTower (output_spatial={args.output_spatial}) ...")
    tower = WanT2VTower(
        checkpoint_dir=args.checkpoint_dir,
        output_spatial=args.output_spatial,
    ).to(args.device).eval()

    img = torch.zeros(1, 3, 224, 224, device=args.device)
    with torch.no_grad():
        feats = tower.encode(img)

    print()
    print(f"  Tower output shape : {tuple(feats.shape)}")
    print(f"  Tower output dtype : {feats.dtype}")
    print(f"  num_tokens         : {feats.shape[1]}  (expected {args.output_spatial * args.output_spatial})")
    print(f"  feat_dim           : {feats.shape[-1]}")
    print()
    print(f"  → Set `vega3d_tower_feat_dim={feats.shape[-1]}` in your TrainConfig.")


if __name__ == "__main__":
    main()
