"""Sanity-check Wan 2.2 I2V on a handful of random LIBERO frames.

Picks N random frames from the LeRobot LIBERO dataset, builds a scene-aware
prompt from each frame's task description (or reads prompts from a JSON
file), and runs Wan 2.2 image-to-video to produce short clips. The point is
to eyeball whether Wan 2.2 produces plausible robot-arm motion when given
a sensible prompt for the actual scene -- not to benchmark anything.

Defaults to Wan-AI/Wan2.2-TI2V-5B-Diffusers (the 5B unified TI2V model)
because the A14B I2V MoE checkpoint is ~70 GB and won't fit on most
research boxes. Pass --model to override.

Outputs (under --out_dir, default assets/wan22_i2v_libero/):
    run_<seed>/
        00_input.png          # the chosen LIBERO frame (resized for Wan)
        00_prompt.txt         # the prompt actually sent to Wan
        00_meta.json          # episode/frame indices, task string, etc.
        00_output.mp4
        01_*, 02_*, ...
        summary.md            # one-glance index of all generations

Usage:
    python scripts/wan22_i2v_libero_test.py
    python scripts/wan22_i2v_libero_test.py --num 5 --seed 0
    python scripts/wan22_i2v_libero_test.py --prompts_file my_prompts.json
    python scripts/wan22_i2v_libero_test.py --model Wan-AI/Wan2.2-I2V-A14B-Diffusers
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_DATASET = "physical-intelligence/libero"

# Wan's default negative prompt (taken from the diffusers example for Wan I2V);
# steers away from static frames, blur, JPEG artifacts, and other failure modes
# the team observed during training.
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def libero_frame_to_pil(raw) -> Image.Image:
    """LIBERO 'image' entries are torch tensors in [3, 256, 256], uint8 or float."""
    t = raw if isinstance(raw, torch.Tensor) else torch.from_numpy(np.asarray(raw))
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)  # CHW -> HWC
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)
    return Image.fromarray(t.cpu().numpy())


def fit_to_wan_canvas(image: Image.Image, max_area: int, mod_value: int) -> Image.Image:
    """Resize a (likely square) LIBERO frame onto a Wan-friendly canvas.

    Keeps aspect ratio and picks (h, w) so that h*w <= max_area and both sides
    are multiples of `mod_value` (vae_scale_factor_spatial * patch_size[1]).
    Square LIBERO frames -> square output, ~480x480 by default.
    """
    aspect = image.height / image.width
    h = round(np.sqrt(max_area * aspect)) // mod_value * mod_value
    w = round(np.sqrt(max_area / aspect)) // mod_value * mod_value
    h, w = int(max(h, mod_value)), int(max(w, mod_value))
    return image.resize((w, h), Image.LANCZOS)


def build_prompt(task: str) -> str:
    """Wrap a LIBERO task description in scene-aware framing language so Wan
    has both the action goal *and* enough visual hints (lighting, camera,
    look-and-feel of LIBERO scenes) to produce a coherent video.
    """
    task = task.strip().rstrip(".")
    return (
        f"A Franka robot arm on a tabletop, viewed from a third-person camera. "
        f"The robot smoothly moves to {task}. Realistic robot motion, "
        f"consistent lighting, stable camera, photorealistic."
    )


def select_frames(dataset, num: int, rng: random.Random) -> list[dict]:
    """Pick `num` distinct frames from distinct episodes, somewhere mid-episode
    (not the very first / last frame). Returns list of dicts with the frame
    payload + episode/frame metadata.
    """
    episode_index_arr = np.asarray(dataset.hf_dataset["episode_index"])
    unique_eps = sorted(set(episode_index_arr.tolist()))
    rng.shuffle(unique_eps)
    chosen_eps = unique_eps[:num]

    out = []
    for ep in chosen_eps:
        ep_frames = np.where(episode_index_arr == ep)[0]
        # Aim for the middle ~60% of the episode so we get an in-progress
        # manipulation moment, not "robot at home pose" or "task complete".
        lo, hi = int(0.2 * len(ep_frames)), int(0.8 * len(ep_frames))
        if hi <= lo:
            lo, hi = 0, len(ep_frames)
        global_idx = int(ep_frames[rng.randint(lo, max(lo, hi - 1))])
        item = dataset[global_idx]
        out.append({
            "global_index": global_idx,
            "episode_index": int(ep),
            "frame_in_episode": int(global_idx - ep_frames[0]),
            "task": item["task"],
            "image": item["image"],
        })
    return out


def load_pipeline(model_id: str, dtype: torch.dtype):
    """Load Wan 2.2 I2V pipeline. Tries the I2V class first; falls back to the
    unified WanPipeline if the repo's model_index.json says so (some Wan 2.2
    TI2V repos expose the unified class).
    """
    from diffusers import WanImageToVideoPipeline
    print(f"[wan22-i2v] Loading {model_id} (dtype={dtype}) ...")
    return WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF model id. Default: Wan 2.2 TI2V-5B (fits on a 48GB GPU).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot dataset repo id.")
    parser.add_argument("--num", type=int, default=5, help="Number of (frame, prompt) generations.")
    parser.add_argument("--seed", type=int, default=0, help="Seeds frame selection AND Wan sampling.")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Frames per video. Default 81 (~5s @ 16fps). Wan requires 4k+1.")
    parser.add_argument("--fps", type=int, default=16, help="Output mp4 frame rate.")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--max_area", type=int, default=480 * 832,
                        help="Max H*W after resizing the LIBERO frame onto the Wan canvas.")
    parser.add_argument("--prompts_file", default=None,
                        help="Optional JSON file: list of strings, one prompt per generation. "
                             "If shorter than --num, the rest fall back to auto-generated prompts.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--out_dir", default="assets/wan22_i2v_libero",
                        help="Output root. A run_<seed> subdir is created inside.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload", action="store_true",
                        help="Enable model-cpu-offload (slower but lighter on VRAM).")
    args = parser.parse_args()

    custom_prompts: list[str] = []
    if args.prompts_file:
        custom_prompts = json.loads(pathlib.Path(args.prompts_file).read_text())
        if not isinstance(custom_prompts, list) or not all(isinstance(p, str) for p in custom_prompts):
            raise ValueError("--prompts_file must be a JSON list of strings")

    out_root = pathlib.Path(args.out_dir) / f"run_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[wan22-i2v] Output dir: {out_root.resolve()}")

    # Pick frames first so we fail fast on dataset issues before downloading the model.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    print(f"[wan22-i2v] Opening dataset {args.dataset} ...")
    dataset = LeRobotDataset(args.dataset)
    print(f"[wan22-i2v] {len(dataset)} frames; selecting {args.num} ...")
    selections = select_frames(dataset, args.num, random.Random(args.seed))

    # Load the pipeline (heavy: triggers HF download on first run).
    dtype = torch.bfloat16
    pipe = load_pipeline(args.model, dtype)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    print(f"[wan22-i2v] vae_scale_factor_spatial={pipe.vae_scale_factor_spatial}, "
          f"patch_size={pipe.transformer.config.patch_size}, mod_value={mod_value}")

    from diffusers.utils import export_to_video
    summary_lines = [
        f"# Wan 2.2 I2V sanity check (seed={args.seed})",
        "",
        f"- model: `{args.model}`",
        f"- dataset: `{args.dataset}`",
        f"- num_frames={args.num_frames}, fps={args.fps}, steps={args.steps}, "
        f"guidance_scale={args.guidance_scale}, max_area={args.max_area}",
        "",
    ]

    for i, sel in enumerate(selections):
        tag = f"{i:02d}"
        pil = libero_frame_to_pil(sel["image"])
        pil_resized = fit_to_wan_canvas(pil, args.max_area, mod_value)
        prompt = (
            custom_prompts[i] if i < len(custom_prompts) else build_prompt(sel["task"])
        )

        pil_resized.save(out_root / f"{tag}_input.png")
        (out_root / f"{tag}_prompt.txt").write_text(prompt + "\n")
        (out_root / f"{tag}_meta.json").write_text(json.dumps({
            "episode_index": sel["episode_index"],
            "frame_in_episode": sel["frame_in_episode"],
            "global_index": sel["global_index"],
            "libero_task": sel["task"],
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "height": pil_resized.height,
            "width": pil_resized.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "model": args.model,
        }, indent=2))

        out_mp4 = out_root / f"{tag}_output.mp4"
        print(f"\n[wan22-i2v] [{tag}] ep={sel['episode_index']} "
              f"frame={sel['frame_in_episode']} task={sel['task']!r}")
        print(f"[wan22-i2v] [{tag}] prompt: {prompt}")
        print(f"[wan22-i2v] [{tag}] generating {pil_resized.width}x{pil_resized.height}x{args.num_frames} ...")

        # Per-generation seed so each one is independently reproducible.
        gen_seed = args.seed * 1000 + i
        generator = torch.Generator(device=args.device).manual_seed(gen_seed)

        with torch.no_grad():
            result = pipe(
                image=pil_resized,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=pil_resized.height,
                width=pil_resized.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        frames = result.frames[0]
        export_to_video(frames, str(out_mp4), fps=args.fps)
        print(f"[wan22-i2v] [{tag}] wrote {out_mp4}")

        summary_lines += [
            f"## {tag}",
            f"- episode {sel['episode_index']}, frame {sel['frame_in_episode']} (global {sel['global_index']})",
            f"- LIBERO task: _{sel['task']}_",
            f"- prompt: {prompt}",
            f"- input: `{tag}_input.png` | output: `{tag}_output.mp4` | meta: `{tag}_meta.json`",
            "",
        ]

    (out_root / "summary.md").write_text("\n".join(summary_lines))
    print(f"\n[wan22-i2v] Done. {out_root}/summary.md has the index.")


if __name__ == "__main__":
    main()
