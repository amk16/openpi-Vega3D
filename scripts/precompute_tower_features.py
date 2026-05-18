"""Precompute VEGA-3D tower features for all frames in a LeRobot dataset.

Iterates the dataset referenced by a TrainConfig, runs the configured spatial
tower (VAE or WAN-T2V) over the camera images, and writes per-episode
safetensors.

To survive limited local disk, the script works in an embed -> upload -> delete
loop: it computes features locally, periodically uploads completed episodes to
S3, and deletes them locally to free space. S3 is the source of truth for which
episodes are already done, so the run is resumable across interruptions and
machines -- restart it and it skips whatever is already in S3.

Cache layout (identical local staging dir and S3 prefix):

    <prefix>/
        meta.json
        <camera_name>/
            ep_000000.safetensors   # {"features": [num_frames, num_tokens, feat_dim] bf16}
            ep_000001.safetensors
            ...

Usage:
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp \
        --flush_every_episodes 25 --batch_size 16
    # local-only (no S3 upload), e.g. for a smoke test:
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp \
        --s3_bucket "" --limit_episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import safetensors.torch
import torch
import tqdm

from openpi.training.config import get_config

PROMPT_EMB_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "openpi_vega3d" / "towers" / "wan_prompt_embedding.pt"
)

# Maps the post-LiberoInputs camera names (used in the model's vega3d_cameras
# tuple) to the raw keys in the LeRobot dataset. physical-intelligence/libero
# exposes flat keys: "image" (base camera) and "wrist_image" (left wrist).
LIBERO_CAMERA_TO_DATASET_KEY = {
    "base_0_rgb": "image",
    "left_wrist_0_rgb": "wrist_image",
}

HF_WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
EP_FILE_RE = re.compile(r"ep_(\d+)\.safetensors$")


def ensure_wan_checkpoint(checkpoint_dir: str) -> None:
    """Download Wan2.1-T2V-1.3B via huggingface_hub if missing."""
    if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        return
    from huggingface_hub import snapshot_download
    print(f"[precompute] Downloading {HF_WAN_REPO} to {checkpoint_dir} ...")
    snapshot_download(HF_WAN_REPO, local_dir=checkpoint_dir, local_dir_use_symlinks=False)
    print("[precompute] Download complete.")


def ensure_prompt_embedding() -> None:
    """Generate the T5 prompt embedding the WAN tower needs, if missing."""
    if PROMPT_EMB_PATH.exists():
        return
    print(f"[precompute] Prompt embedding missing at {PROMPT_EMB_PATH}; running export script ...")
    export_script = pathlib.Path(__file__).resolve().parent / "export_wan_prompt_embedding.py"
    subprocess.run([sys.executable, str(export_script)], check=True)


def prepare_image(raw, device: torch.device) -> torch.Tensor:
    """Convert a LeRobot dataset image entry to [1, 3, 224, 224] in [-1, 1]."""
    if isinstance(raw, np.ndarray):
        t = torch.from_numpy(raw)
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
    else:  # torch.Tensor
        t = raw
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
    t = t.unsqueeze(0).to(device)  # [1, 3, H, W]
    if t.shape[-2:] != (224, 224):
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0  # [-1, 1]


def list_completed_episodes_s3(bucket: str, prefix: str, cameras) -> set[int]:
    """Return episode indices whose feature files exist in S3 for ALL cameras.

    S3 is the source of truth for resume: an episode counts as done only when
    every requested camera has uploaded its ep_<idx>.safetensors.
    """
    per_cam: dict[str, set[int]] = {}
    for cam in cameras:
        done: set[int] = set()
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{bucket}/{prefix}/{cam}/"],
            capture_output=True, text=True,
        )
        # Non-zero exit just means the prefix doesn't exist yet -> no episodes.
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts:
                    m = EP_FILE_RE.search(parts[-1])
                    if m:
                        done.add(int(m.group(1)))
        per_cam[cam] = done
    if not per_cam:
        return set()
    return set.intersection(*per_cam.values())


def flush_to_s3(local_root: pathlib.Path, bucket: str, prefix: str) -> None:
    """Upload everything under local_root to S3, then delete local feature
    files to free disk. Raises (keeping local files) if the upload fails."""
    dest = f"s3://{bucket}/{prefix}"
    print(f"[precompute] Uploading {local_root} -> {dest} ...")
    # `aws s3 sync` uploads only new/changed files; no --delete, so episodes
    # already in S3 (and since deleted locally) are left untouched.
    subprocess.run(["aws", "s3", "sync", str(local_root), dest], check=True)
    freed = 0
    for p in local_root.rglob("ep_*.safetensors"):
        freed += p.stat().st_size
        p.unlink()
    print(f"[precompute] Upload OK; freed {freed / 1e9:.1f} GB of local disk")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", help="TrainConfig name (e.g. pi05_libero_lora_wan_precomp)")
    parser.add_argument("--cache_dir", default=None,
                        help="Local staging dir. Defaults to config.data.tower_features_cache_dir.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Override cameras to precompute (default: config.model.vega3d_cameras).")
    parser.add_argument("--limit_episodes", type=int, default=None,
                        help="Process at most this many episodes (for smoke-testing).")
    parser.add_argument("--s3_bucket", default="behavior-challenge",
                        help="S3 bucket for upload + resume. Empty string disables S3 (local-only).")
    parser.add_argument("--s3_prefix", default=None,
                        help="S3 key prefix. Defaults to tower_features/<repo>/<tower_variant>.")
    parser.add_argument("--flush_every_episodes", type=int, default=25,
                        help="Upload to S3 and free local disk every N episodes.")
    args = parser.parse_args()

    config = get_config(args.config_name)
    if not getattr(config.model, "use_vega3d", False):
        raise ValueError(f"Config {args.config_name!r} has use_vega3d=False; nothing to precompute.")

    tower_name = config.model.vega3d_tower_name
    tower_kwargs = dict(config.model.vega3d_tower_kwargs or {})
    tower_kwargs.setdefault("output_spatial", 16)
    cameras = tuple(args.cameras) if args.cameras else tuple(config.model.vega3d_cameras)

    cache_dir = args.cache_dir or getattr(config.data, "tower_features_cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "No cache_dir resolved. Pass --cache_dir or use a TrainConfig whose data "
            "factory exposes tower_features_cache_dir (LeRobotLiberoVegaDataConfig)."
        )
    cache_root = pathlib.Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Auto-download WAN + T5 prompt embedding if needed.
    if tower_name == "wan_t2v":
        ensure_wan_checkpoint(tower_kwargs["checkpoint_dir"])
        ensure_prompt_embedding()

    # Build the tower and probe its actual output shape.
    from openpi_vega3d.towers import TOWER_REGISTRY
    print(f"[precompute] Building tower {tower_name} (kwargs={tower_kwargs}) ...")
    tower = TOWER_REGISTRY[tower_name](**tower_kwargs).to(args.device).eval()
    device = torch.device(args.device)
    with torch.no_grad():
        sample = tower.encode(torch.zeros(1, 3, 224, 224, device=device))
    feat_dim = int(sample.shape[-1])
    num_tokens = int(sample.shape[1])
    output_spatial = tower_kwargs["output_spatial"]
    if num_tokens != output_spatial * output_spatial:
        raise RuntimeError(
            f"Tower returned {num_tokens} tokens but output_spatial={output_spatial} "
            f"implies {output_spatial * output_spatial}. Refusing an inconsistent cache."
        )
    declared = getattr(config.model, "vega3d_tower_feat_dim", None)
    if declared is not None and declared != feat_dim:
        print(f"[WARN] config.model.vega3d_tower_feat_dim={declared} but tower outputs "
              f"feat_dim={feat_dim}. Update the TrainConfig to {feat_dim} before training.")

    # Open the dataset (no delta_timestamps -- we want per-frame items).
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    print(f"[precompute] Opening dataset {config.data.repo_id} ...")
    dataset = LeRobotDataset(config.data.repo_id)
    episode_indices_arr = np.asarray(dataset.hf_dataset["episode_index"])
    unique_eps = np.unique(episode_indices_arr).tolist()
    if args.limit_episodes is not None:
        unique_eps = unique_eps[: args.limit_episodes]

    for cam in cameras:
        if cam not in LIBERO_CAMERA_TO_DATASET_KEY:
            raise ValueError(
                f"Don't know how to source camera {cam!r} from a libero dataset. "
                f"Known: {sorted(LIBERO_CAMERA_TO_DATASET_KEY)}"
            )
        (cache_root / cam).mkdir(parents=True, exist_ok=True)

    # Resolve S3 destination and the set of already-completed episodes.
    use_s3 = bool(args.s3_bucket)
    repo_sanitized = config.data.repo_id.replace("/", "_")
    s3_prefix = args.s3_prefix or f"tower_features/{repo_sanitized}/{tower_name}_{output_spatial}x{feat_dim}"

    if use_s3:
        print(f"[precompute] Checking S3 for completed episodes (s3://{args.s3_bucket}/{s3_prefix}) ...")
        completed = list_completed_episodes_s3(args.s3_bucket, s3_prefix, cameras)
    else:
        completed = {
            ep for ep in unique_eps
            if all((cache_root / cam / f"ep_{ep:06d}.safetensors").exists() for cam in cameras)
        }
    todo = [ep for ep in unique_eps if ep not in completed]
    print(f"[precompute] {len(unique_eps)} episodes total | {len(completed)} done | {len(todo)} to do")
    print(f"[precompute] cameras={cameras} feat_dim={feat_dim} "
          f"dest={'s3://' + args.s3_bucket + '/' + s3_prefix if use_s3 else cache_root}")

    # meta.json -- single source of truth for this cache's geometry.
    meta = {
        "tower_name": tower_name,
        "tower_kwargs": tower_kwargs,
        "cameras": list(cameras),
        "feat_dim": feat_dim,
        "output_spatial": output_spatial,
        "num_tokens": num_tokens,
        "dtype": "bfloat16",
        "repo_id": config.data.repo_id,
        "total_frames_in_dataset": int(len(episode_indices_arr)),
        "s3_uri": f"s3://{args.s3_bucket}/{s3_prefix}" if use_s3 else None,
    }
    (cache_root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # embed -> upload -> delete loop.
    since_flush = 0
    for ep in tqdm.tqdm(todo, desc="episodes"):
        frame_indices = np.where(episode_indices_arr == ep)[0].tolist()
        per_cam_features: dict[str, list[torch.Tensor]] = {cam: [] for cam in cameras}

        for batch_start in range(0, len(frame_indices), args.batch_size):
            batch_indices = frame_indices[batch_start: batch_start + args.batch_size]
            # Load each frame once; collect per-camera image batches.
            cam_imgs: dict[str, list[torch.Tensor]] = {cam: [] for cam in cameras}
            for fi in batch_indices:
                item = dataset[fi]
                for cam in cameras:
                    cam_imgs[cam].append(prepare_image(item[LIBERO_CAMERA_TO_DATASET_KEY[cam]], device))
            for cam in cameras:
                with torch.no_grad():
                    feats = tower.encode(torch.cat(cam_imgs[cam], dim=0))
                per_cam_features[cam].append(feats.detach().to(torch.bfloat16).cpu())

        for cam in cameras:
            cat = torch.cat(per_cam_features[cam], dim=0).contiguous()
            assert cat.shape[0] == len(frame_indices), (
                f"Frame count mismatch for ep {ep} cam {cam}: {cat.shape[0]} vs {len(frame_indices)}"
            )
            out_path = cache_root / cam / f"ep_{ep:06d}.safetensors"
            tmp_path = out_path.with_suffix(".safetensors.tmp")
            safetensors.torch.save_file({"features": cat}, str(tmp_path))
            tmp_path.rename(out_path)

        since_flush += 1
        if use_s3 and since_flush >= args.flush_every_episodes:
            flush_to_s3(cache_root, args.s3_bucket, s3_prefix)
            since_flush = 0

    # Final flush for the trailing partial batch of episodes.
    if use_s3 and since_flush > 0:
        flush_to_s3(cache_root, args.s3_bucket, s3_prefix)

    dest = f"s3://{args.s3_bucket}/{s3_prefix}" if use_s3 else str(cache_root)
    print(f"[precompute] Done. Features at {dest}")


if __name__ == "__main__":
    main()
