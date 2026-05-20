"""Probe DreamDojo (Cosmos-Predict2.5-2B) output shape for vega3d_tower_feat_dim.

Builds the DreamDojoTower at output_spatial=16 and input_resolution=256,
runs a single image through, and prints the actual output shape. Works in
offline mode (no checkpoint) to verify config values, or online mode with
a real checkpoint to verify end-to-end.

Usage:
    python scripts/probe_dreamdojo.py
    python scripts/probe_dreamdojo.py --checkpoint_dir /path/to/DreamDojo-2B
"""

import argparse
import os

import torch

DEFAULT_CKPT_DIR = "/workspace/openpi-Vega3D/ckpts/DreamDojo-2B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--vae_dir", default=None,
                        help="Path to Cosmos VAE directory. If omitted, looks for "
                             "<checkpoint_dir>/vae/ or $COSMOS_VAE_DIR.")
    parser.add_argument("--output_spatial", type=int, default=16)
    parser.add_argument("--input_resolution", type=int, default=256)
    parser.add_argument("--feat_block_idx", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    has_ckpt = os.path.isdir(args.checkpoint_dir) and any(
        f.endswith(".pt") for f in os.listdir(args.checkpoint_dir)
    ) if os.path.isdir(args.checkpoint_dir) else False

    if not has_ckpt:
        print(f"[probe] No checkpoint at {args.checkpoint_dir} — running in offline mode.")
        print("[probe] To run online, download DreamDojo 2B pretrain:")
        print("        1. Clone nvidia/DreamDojo from HuggingFace")
        print("        2. Convert DCP to .pt via cosmos-predict2.5's convert_distcp_to_pt.py")
        print(f"        3. Place .pt file in {args.checkpoint_dir}/")
        print()

    from openpi_vega3d.towers.dreamdojo_tower import DreamDojoTower

    print(f"[probe] Building DreamDojoTower (output_spatial={args.output_spatial}, "
          f"input_resolution={args.input_resolution}, feat_block_idx={args.feat_block_idx}) ...")
    tower = DreamDojoTower(
        checkpoint_dir=args.checkpoint_dir,
        vae_dir=args.vae_dir,
        output_spatial=args.output_spatial,
        input_resolution=args.input_resolution,
        feat_block_idx=args.feat_block_idx,
    ).to(args.device).eval()

    img = torch.zeros(1, 3, 224, 224, device=args.device)
    with torch.no_grad():
        feats = tower.encode(img)

    online_str = "ONLINE (real weights)" if tower.online else "OFFLINE (dummy zeros)"

    print()
    print(f"  Mode               : {online_str}")
    print(f"  Tower output shape : {tuple(feats.shape)}")
    print(f"  Tower output dtype : {feats.dtype}")
    print(f"  num_tokens         : {feats.shape[1]}  (expected {args.output_spatial * args.output_spatial})")
    print(f"  feat_dim           : {feats.shape[-1]}")
    print()
    print(f"  → Set `vega3d_tower_feat_dim={feats.shape[-1]}` in your TrainConfig.")

    if not tower.online:
        print()
        print("  ⚠ Offline mode: output is zeros. Download checkpoint for real features.")


if __name__ == "__main__":
    main()
