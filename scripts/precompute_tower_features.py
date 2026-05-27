"""Precompute VEGA-3D tower features for all frames in a LeRobot dataset.

Iterates the dataset referenced by a TrainConfig, runs the configured spatial
tower (VAE or WAN-T2V) over a temporal window of frames ending at each training
frame, and writes per-episode safetensors. Single-frame mode (window=1) gives
paper-faithful per-frame extraction; window>1 bundles the window as ONE WAN
clip so the DiT's cross-frame attention actually runs, and we keep the last
latent slot (causal summary of the recent past at the training frame). The
output cache shape is identical in both modes -- only the input changes.

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
    # Single-frame (paper-faithful):
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp --window 1

    # Multi-frame window (beyond-paper, gives temporal/dynamics signal for VLAs):
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp \
        --window 17 --stride 2 --batch_size 4

    # local-only (no S3 upload), e.g. for a smoke test:
    python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp \
        --s3_bucket "" --limit_episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading

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
HF_COSMOS_REPO = "nvidia/Cosmos-Predict2.5-2B"
HF_COSMOS_POLICY_LIBERO_REPO = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
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


def ensure_dreamdojo_checkpoint(checkpoint_dir: str) -> None:
    """Check that a DreamDojo .pt checkpoint exists; print guidance if missing."""
    if os.path.isdir(checkpoint_dir) and any(f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
        return
    raise FileNotFoundError(
        f"DreamDojo checkpoint not found at {checkpoint_dir}. To prepare it:\n"
        "  1. Download DreamDojo 2B pretrain from nvidia/DreamDojo on HuggingFace\n"
        "     (2B_pretrain/iter_000140000/model/ directory)\n"
        "  2. Convert DCP to .pt: python convert_distcp_to_pt.py <dcp_dir> <output.pt>\n"
        f"  3. Place the .pt file in {checkpoint_dir}/\n"
        "  4. Place Cosmos VAE in <checkpoint_dir>/vae/ or set $COSMOS_VAE_DIR"
    )


def ensure_cosmos_base_checkpoint(checkpoint_dir: str) -> None:
    """Check that a base Cosmos .pt checkpoint exists; print guidance if missing."""
    if os.path.isdir(checkpoint_dir) and any(f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
        return
    raise FileNotFoundError(
        f"Base Cosmos checkpoint not found at {checkpoint_dir}. To prepare it:\n"
        f"  1. Accept the license at https://huggingface.co/{HF_COSMOS_REPO}\n"
        "  2. Download the post-trained checkpoint:\n"
        f"     huggingface-cli download {HF_COSMOS_REPO} base/post-trained/ "
        f"--local-dir {checkpoint_dir}\n"
        f"  3. Move the .pt file to {checkpoint_dir}/ (top level)\n"
        "  4. Place Cosmos VAE in <checkpoint_dir>/vae/ or set $COSMOS_VAE_DIR\n"
        "     (same VAE as DreamDojo — can symlink from DreamDojo-2B/vae/)"
    )


def ensure_cosmos_libero_checkpoint(checkpoint_dir: str) -> None:
    """Check that a Cosmos LIBERO (Robot/Policy) .pt checkpoint exists."""
    if os.path.isdir(checkpoint_dir) and any(f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
        return
    raise FileNotFoundError(
        f"Cosmos LIBERO checkpoint not found at {checkpoint_dir}. To prepare it:\n"
        f"  1. Accept the license at https://huggingface.co/{HF_COSMOS_REPO}\n"
        "  2. Download the Robot/Policy/Libero checkpoint:\n"
        f"     huggingface-cli download {HF_COSMOS_REPO} robot/policy/libero/model.pt "
        f"--local-dir {checkpoint_dir}\n"
        f"  3. Move model.pt to {checkpoint_dir}/ (top level)\n"
        "  4. Place Cosmos VAE (AutoencoderKLWan, diffusers format) in\n"
        f"     {checkpoint_dir}/vae/ or set $COSMOS_VAE_DIR"
    )


def ensure_cosmos_policy_libero_checkpoint(checkpoint_dir: str) -> None:
    """Check that a Cosmos-Policy-LIBERO .pt checkpoint exists."""
    if os.path.isdir(checkpoint_dir) and any(f.endswith(".pt") for f in os.listdir(checkpoint_dir)):
        return
    raise FileNotFoundError(
        f"Cosmos-Policy-LIBERO checkpoint not found at {checkpoint_dir}. To prepare it:\n"
        f"  1. Accept the license at https://huggingface.co/{HF_COSMOS_POLICY_LIBERO_REPO}\n"
        f"  2. huggingface-cli download {HF_COSMOS_POLICY_LIBERO_REPO} "
        f"--local-dir {checkpoint_dir}\n"
        f"  3. Copy the .pt to {checkpoint_dir}/model.pt\n"
        "  4. Place Cosmos VAE (AutoencoderKLWan, diffusers format) in\n"
        f"     {checkpoint_dir}/vae/ or set $COSMOS_VAE_DIR"
    )


def prepare_image(raw, device: torch.device, resolution: int = 224) -> torch.Tensor:
    """Convert a LeRobot dataset image entry to [1, 3, H, W] in [-1, 1]."""
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
    target = (resolution, resolution)
    if t.shape[-2:] != target:
        t = torch.nn.functional.interpolate(t, size=target, mode="bilinear", align_corners=False)
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


def flush_to_s3(
    local_root: pathlib.Path,
    bucket: str,
    prefix: str,
    *,
    keep_local: bool = False,
) -> None:
    """Upload everything under local_root to S3.

    By default also deletes local feature files after upload to free disk.
    Pass `keep_local=True` to retain local files (e.g. so a subsequent
    training run can read them without re-downloading). Raises if the
    upload fails -- local files are kept on error regardless of flag.
    """
    dest = f"s3://{bucket}/{prefix}"
    print(f"[precompute] Uploading {local_root} -> {dest} ...")
    # `aws s3 sync` uploads only new/changed files; no --delete, so episodes
    # already in S3 (and since deleted locally) are left untouched.
    subprocess.run(["aws", "s3", "sync", str(local_root), dest], check=True)
    if keep_local:
        print("[precompute] Upload OK; keeping local files (--keep_local_after_upload)")
        return
    freed = 0
    for p in local_root.rglob("ep_*.safetensors"):
        freed += p.stat().st_size
        p.unlink()
    print(f"[precompute] Upload OK; freed {freed / 1e9:.1f} GB of local disk")


def _prefetch_iter(iterable, maxsize: int = 3):
    """Wrap an iterable with a background-thread prefetch queue."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    _sentinel = object()

    def _produce():
        try:
            for item in iterable:
                q.put(item)
        finally:
            q.put(_sentinel)

    t = threading.Thread(target=_produce, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _sentinel:
            break
        yield item
    t.join()


def _generate_batches(
    todo, episode_indices_arr, dataset, cameras, cameras_list,
    batch_size, window, stride, image_resolution, ep_to_text_embed,
):
    """Yield pre-loaded batch dicts (all tensors on CPU) for the GPU loop."""
    cpu = torch.device("cpu")
    for ep in todo:
        frame_indices = np.where(episode_indices_arr == ep)[0].tolist()
        n_frames = len(frame_indices)
        ep_text_embed = ep_to_text_embed.get(ep)

        batch_starts = list(range(0, n_frames, batch_size))
        for bi, batch_start_pos in enumerate(batch_starts):
            batch_positions = list(range(
                batch_start_pos, min(batch_start_pos + batch_size, n_frames),
            ))
            bs = len(batch_positions)

            text_embed = None
            if ep_text_embed is not None:
                text_embed = ep_text_embed.unsqueeze(0).expand(bs, -1, -1)

            window_positions_per_item = [
                build_window_positions(p, window, stride) for p in batch_positions
            ]
            unique_positions = sorted({pos for wp in window_positions_per_item for pos in wp})
            prepared_by_pos: dict[int, dict[str, torch.Tensor]] = {}
            for pos in unique_positions:
                item = dataset[frame_indices[pos]]
                prepared_by_pos[pos] = {
                    cam: prepare_image(
                        item[LIBERO_CAMERA_TO_DATASET_KEY[cam]], cpu, image_resolution,
                    ).squeeze(0)
                    for cam in cameras
                }

            clips_per_cam = {}
            seeds_per_cam = {}
            for cam in cameras:
                clip_list = [
                    torch.stack([prepared_by_pos[pos][cam] for pos in wp], dim=0)
                    for wp in window_positions_per_item
                ]
                clips_per_cam[cam] = torch.stack(clip_list, dim=0)
                cam_idx = cameras_list.index(cam)
                seeds_per_cam[cam] = (
                    int(ep) * 1_000_003 + int(batch_start_pos) * 17 + cam_idx
                ) % (2**31 - 1)

            yield {
                "ep": ep,
                "n_frames": n_frames,
                "is_last": bi == len(batch_starts) - 1,
                "clips_per_cam": clips_per_cam,
                "text_embed": text_embed,
                "seeds_per_cam": seeds_per_cam,
            }


def _generate_policy_batches(
    todo, episode_indices_arr, dataset, cameras, batch_size,
    image_resolution, ep_to_text_embed,
):
    """Yield batches for policy tower: both cameras + proprio per frame."""
    cpu = torch.device("cpu")
    wrist_key = LIBERO_CAMERA_TO_DATASET_KEY["left_wrist_0_rgb"]
    primary_key = LIBERO_CAMERA_TO_DATASET_KEY["base_0_rgb"]
    for ep in todo:
        frame_indices = np.where(episode_indices_arr == ep)[0].tolist()
        n_frames = len(frame_indices)
        ep_text_embed = ep_to_text_embed.get(ep)

        batch_starts = list(range(0, n_frames, batch_size))
        for bi, batch_start_pos in enumerate(batch_starts):
            batch_positions = list(range(
                batch_start_pos, min(batch_start_pos + batch_size, n_frames),
            ))
            bs = len(batch_positions)

            text_embed = None
            if ep_text_embed is not None:
                text_embed = ep_text_embed.unsqueeze(0).expand(bs, -1, -1)

            wrist_imgs = []
            primary_imgs = []
            for pos in batch_positions:
                item = dataset[frame_indices[pos]]
                wrist_imgs.append(
                    prepare_image(item[wrist_key], cpu, image_resolution).squeeze(0)
                )
                primary_imgs.append(
                    prepare_image(item[primary_key], cpu, image_resolution).squeeze(0)
                )

            seed = (int(ep) * 1_000_003 + int(batch_start_pos) * 17) % (2**31 - 1)

            yield {
                "ep": ep,
                "n_frames": n_frames,
                "is_last": bi == len(batch_starts) - 1,
                "wrist_imgs": torch.stack(wrist_imgs, dim=0),
                "primary_imgs": torch.stack(primary_imgs, dim=0),
                "text_embed": text_embed,
                "seed": seed,
            }


def build_window_positions(p: int, window: int, stride: int) -> list[int]:
    """Causal window of within-episode positions ending at `p`.

    Returns [p - stride*(W-1), ..., p - stride, p], with any position < 0
    clamped to 0 (i.e., repeat the episode's first frame at the start). Never
    reaches across episode boundaries.
    """
    positions = []
    for k in range(window - 1, -1, -1):
        pos = p - stride * k
        if pos < 0:
            pos = 0
        positions.append(pos)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", help="TrainConfig name (e.g. pi05_libero_lora_wan_precomp)")
    parser.add_argument("--cache_dir", default=None,
                        help="Local staging dir. Defaults to config.data.tower_features_cache_dir.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of windows per WAN forward. Multi-frame is heavier; lower this "
                             "if you OOM. 4 is a safe default for T=17 on a 48GB GPU.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Override cameras to precompute (default: config.model.vega3d_cameras).")
    parser.add_argument("--limit_episodes", type=int, default=None,
                        help="Process at most this many episodes (for smoke-testing).")
    parser.add_argument("--start_episode", type=int, default=None,
                        help="Start from this episode index (inclusive). For multi-GPU: split "
                             "the episode list across processes.")
    parser.add_argument("--end_episode", type=int, default=None,
                        help="Stop before this episode index (exclusive). Combine with "
                             "--start_episode to shard across GPUs.")
    parser.add_argument("--s3_bucket", default="behavior-challenge",
                        help="S3 bucket for upload + resume. Empty string disables S3 (local-only).")
    parser.add_argument("--s3_prefix", default=None,
                        help="S3 key prefix. Default bakes in tower variant + window/stride + block "
                             "idx so different settings produce different caches.")
    parser.add_argument("--flush_every_episodes", type=int, default=25,
                        help="Upload to S3 and free local disk every N episodes.")
    parser.add_argument("--keep_local_after_upload", action="store_true", default=False,
                        help="Keep local feature files after S3 upload instead of deleting "
                             "them. Useful when a downstream training run on the same machine "
                             "needs the cache on disk -- skips the round-trip through S3. "
                             "WARNING: full LIBERO Cosmos cache is ~534 GiB; ensure free disk.")
    parser.add_argument("--window", type=int, default=None,
                        help="Temporal window size (frames per WAN clip). For each training frame f, "
                             "features come from frames [f - stride*(W-1) ... f], clamped at episode "
                             "start. window=1 is paper-faithful per-frame extraction. window>1 "
                             "activates WAN's cross-frame attention. Default: config.data.tower_window "
                             "or 1.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Frame stride within the window. e.g. window=17 stride=2 covers 33 real "
                             "frames of motion (~1.6s at 20Hz). Default: config.data.tower_stride or 1.")
    parser.add_argument("--prompt_cache", default=None,
                        help="Path to precomputed T5 prompt embeddings (.pt) from "
                             "export_cosmos_prompt_embeddings.py. When provided, each episode's "
                             "task prompt is looked up and passed as text conditioning to the tower.")
    args = parser.parse_args()

    config = get_config(args.config_name)
    if not getattr(config.model, "use_vega3d", False):
        raise ValueError(f"Config {args.config_name!r} has use_vega3d=False; nothing to precompute.")

    tower_name = config.model.vega3d_tower_name
    tower_kwargs = dict(config.model.vega3d_tower_kwargs or {})
    tower_kwargs.setdefault("output_spatial", 16)
    cameras = tuple(args.cameras) if args.cameras else tuple(config.model.vega3d_cameras)

    # Resolve window / stride: CLI > config > 1.
    window = args.window if args.window is not None else int(getattr(config.data, "tower_window", 1) or 1)
    stride = args.stride if args.stride is not None else int(getattr(config.data, "tower_stride", 1) or 1)
    if window < 1 or stride < 1:
        raise ValueError(f"window and stride must be >= 1; got window={window} stride={stride}")
    multi_frame = window > 1
    print(f"[precompute] window={window} stride={stride} "
          f"({'multi-frame (cross-frame attn ON)' if multi_frame else 'single-frame (paper-faithful)'})")

    cache_dir = args.cache_dir or getattr(config.data, "tower_features_cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "No cache_dir resolved. Pass --cache_dir or use a TrainConfig whose data "
            "factory exposes tower_features_cache_dir (LeRobotLiberoVegaDataConfig)."
        )
    cache_root = pathlib.Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Auto-download / verify tower checkpoint.
    if tower_name == "wan_t2v":
        ensure_wan_checkpoint(tower_kwargs["checkpoint_dir"])
        ensure_prompt_embedding()
    elif tower_name == "dreamdojo":
        ensure_dreamdojo_checkpoint(tower_kwargs["checkpoint_dir"])
    elif tower_name == "cosmos_base":
        ensure_cosmos_base_checkpoint(tower_kwargs["checkpoint_dir"])
    elif tower_name == "cosmos_libero":
        ensure_cosmos_libero_checkpoint(tower_kwargs["checkpoint_dir"])
    elif tower_name == "cosmos_policy_libero":
        ensure_cosmos_policy_libero_checkpoint(tower_kwargs["checkpoint_dir"])

    # Resolve image resolution for prepare_image(). DreamDojo/Cosmos use 256x256
    # internally; feeding that directly avoids a redundant 224→256 resize.
    image_resolution = int(tower_kwargs.get("input_resolution", 224))

    # Build the tower and probe its actual output shape.
    from openpi_vega3d.towers import TOWER_REGISTRY
    print(f"[precompute] Building tower {tower_name} (kwargs={tower_kwargs}) ...")
    tower = TOWER_REGISTRY[tower_name](**tower_kwargs).to(args.device).eval()
    device = torch.device(args.device)

    is_policy_tower = tower_name == "cosmos_policy_libero"

    # Probe via the same code path we'll use for real.
    with torch.no_grad():
        if is_policy_tower:
            probe_img = torch.zeros(1, 3, image_resolution, image_resolution, device=device)
            sample, _ = tower.encode_policy_batch(probe_img, probe_img, proprio=None, noise_seed=0)
        else:
            probe_clips = torch.zeros(1, max(window, 1), 3, image_resolution, image_resolution, device=device)
            sample = tower.encode_window_batch(probe_clips, noise_seed=0)
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
    if args.start_episode is not None:
        unique_eps = [ep for ep in unique_eps if ep >= args.start_episode]
    if args.end_episode is not None:
        unique_eps = [ep for ep in unique_eps if ep < args.end_episode]
    if args.limit_episodes is not None:
        unique_eps = unique_eps[: args.limit_episodes]

    # Prompt cache: load precomputed T5 embeddings and build episode → embedding map.
    prompt_cache = None
    ep_to_text_embed: dict[int, torch.Tensor] = {}
    if args.prompt_cache:
        from openpi_vega3d.towers.prompt_cache import PromptEmbeddingCache
        prompt_cache = PromptEmbeddingCache(args.prompt_cache)
        print(f"[precompute] Loaded prompt cache with {len(prompt_cache)} prompts")

        task_index_arr = np.asarray(dataset.hf_dataset["task_index"])
        task_map = dataset.meta.tasks  # {int: str}
        for ep in unique_eps:
            first_frame = np.where(episode_indices_arr == ep)[0][0]
            task_idx = int(task_index_arr[first_frame])
            task_str = task_map[task_idx]
            ep_to_text_embed[ep] = prompt_cache[task_str]  # [seq_len, embed_dim]
        unique_prompts_used = len({id(v) for v in ep_to_text_embed.values()})
        print(f"[precompute] Mapped {len(ep_to_text_embed)} episodes to {unique_prompts_used} unique prompts")

    for cam in cameras:
        if cam not in LIBERO_CAMERA_TO_DATASET_KEY:
            raise ValueError(
                f"Don't know how to source camera {cam!r} from a libero dataset. "
                f"Known: {sorted(LIBERO_CAMERA_TO_DATASET_KEY)}"
            )
        (cache_root / cam).mkdir(parents=True, exist_ok=True)

    # Resolve S3 destination and the set of already-completed episodes. Bake
    # window/stride/feat_block_idx into the default prefix so different
    # geometries don't silently overwrite each other.
    use_s3 = bool(args.s3_bucket)
    repo_sanitized = config.data.repo_id.replace("/", "_")
    feat_block_idx = int(tower_kwargs.get("feat_block_idx", -1))
    text_tag = "_t5cond" if args.prompt_cache else ""
    variant_tag = f"{tower_name}_{output_spatial}x{feat_dim}_w{window}s{stride}_blk{feat_block_idx}{text_tag}"
    s3_prefix = args.s3_prefix or f"tower_features/{repo_sanitized}/{variant_tag}"

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
        # Window / stride / block: how this cache was built. The model side is
        # invariant to these (it just reads [N, num_tokens, feat_dim] per frame),
        # but the *meaning* of the features changes -- record it for posterity.
        "window": window,
        "stride": stride,
        "feat_block_idx": feat_block_idx,
        "extraction_mode": "multi_frame_last_slot" if multi_frame else "single_frame",
        "text_conditioned": args.prompt_cache is not None,
    }
    (cache_root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # embed -> upload -> delete loop, with background data prefetch.
    since_flush = 0
    cameras_list = list(cameras)

    if is_policy_tower:
        # Policy tower: process both cameras jointly via encode_policy_batch.
        batch_gen = _generate_policy_batches(
            todo, episode_indices_arr, dataset, cameras, args.batch_size,
            image_resolution, ep_to_text_embed,
        )
        per_cam_features: dict[str, list[torch.Tensor]] = {cam: [] for cam in cameras}
        ep_bar = tqdm.tqdm(total=len(todo), desc="episodes")

        for batch in _prefetch_iter(batch_gen, maxsize=3):
            ep = batch["ep"]
            text_embed = batch["text_embed"].to(device) if batch["text_embed"] is not None else None
            wrist_imgs = batch["wrist_imgs"].to(device)
            primary_imgs = batch["primary_imgs"].to(device)

            with torch.no_grad():
                wrist_feats, primary_feats = tower.encode_policy_batch(
                    wrist_imgs, primary_imgs, proprio=None,
                    text_embed=text_embed, noise_seed=batch["seed"],
                )
            per_cam_features["left_wrist_0_rgb"].append(wrist_feats.detach().to(torch.bfloat16).cpu())
            per_cam_features["base_0_rgb"].append(primary_feats.detach().to(torch.bfloat16).cpu())

            if batch["is_last"]:
                n_frames = batch["n_frames"]
                for cam in cameras:
                    cat = torch.cat(per_cam_features[cam], dim=0).contiguous()
                    assert cat.shape[0] == n_frames, (
                        f"Frame count mismatch for ep {ep} cam {cam}: {cat.shape[0]} vs {n_frames}"
                    )
                    out_path = cache_root / cam / f"ep_{ep:06d}.safetensors"
                    tmp_path = out_path.with_suffix(".safetensors.tmp")
                    safetensors.torch.save_file({"features": cat}, str(tmp_path))
                    tmp_path.rename(out_path)
                per_cam_features = {cam: [] for cam in cameras}
                ep_bar.update(1)

                since_flush += 1
                if use_s3 and since_flush >= args.flush_every_episodes:
                    flush_to_s3(cache_root, args.s3_bucket, s3_prefix,
                                keep_local=args.keep_local_after_upload)
                    since_flush = 0

        ep_bar.close()

    else:
        # Standard tower: process cameras independently.
        batch_gen = _generate_batches(
            todo, episode_indices_arr, dataset, cameras, cameras_list,
            args.batch_size, window, stride, image_resolution, ep_to_text_embed,
        )
        per_cam_features = {cam: [] for cam in cameras}
        ep_bar = tqdm.tqdm(total=len(todo), desc="episodes")

        for batch in _prefetch_iter(batch_gen, maxsize=3):
            ep = batch["ep"]
            text_embed = batch["text_embed"].to(device) if batch["text_embed"] is not None else None

            for cam in cameras:
                clips = batch["clips_per_cam"][cam].to(device)
                with torch.no_grad():
                    feats = tower.encode_window_batch(
                        clips, noise_seed=batch["seeds_per_cam"][cam], text_embed=text_embed,
                    )
                per_cam_features[cam].append(feats.detach().to(torch.bfloat16).cpu())

            if batch["is_last"]:
                n_frames = batch["n_frames"]
                for cam in cameras:
                    cat = torch.cat(per_cam_features[cam], dim=0).contiguous()
                    assert cat.shape[0] == n_frames, (
                        f"Frame count mismatch for ep {ep} cam {cam}: {cat.shape[0]} vs {n_frames}"
                    )
                    out_path = cache_root / cam / f"ep_{ep:06d}.safetensors"
                    tmp_path = out_path.with_suffix(".safetensors.tmp")
                    safetensors.torch.save_file({"features": cat}, str(tmp_path))
                    tmp_path.rename(out_path)
                per_cam_features = {cam: [] for cam in cameras}
                ep_bar.update(1)

                since_flush += 1
                if use_s3 and since_flush >= args.flush_every_episodes:
                    flush_to_s3(cache_root, args.s3_bucket, s3_prefix,
                                keep_local=args.keep_local_after_upload)
                    since_flush = 0

        ep_bar.close()

    # Final flush for the trailing partial batch of episodes.
    if use_s3 and since_flush > 0:
        flush_to_s3(cache_root, args.s3_bucket, s3_prefix,
                    keep_local=args.keep_local_after_upload)

    dest = f"s3://{args.s3_bucket}/{s3_prefix}" if use_s3 else str(cache_root)
    print(f"[precompute] Done. Features at {dest}")


if __name__ == "__main__":
    main()
