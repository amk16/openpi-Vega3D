"""Export a T5 prompt embedding for the WAN-T2V vision tower.

WAN-T2V is a text-to-video diffusion model. Its cross-attention layers expect a
text context tensor at every forward pass, even when we're only using it as a
vision feature encoder (which is what `WanT2VOnlineEncoder` does). We avoid
running T5 inside the training/precompute loop by encoding ONE prompt once and
saving the resulting [seq_len, hidden] tensor to disk; the tower loads it lazily
and reuses it for every image batch.

This script uses HuggingFace's UMT5EncoderModel (google/umt5-xxl), which
matches Wan2.1's text encoder architecture. The first run downloads ~11 GB of
T5 encoder weights from HF.

Output (default): src/openpi_vega3d/towers/wan_prompt_embedding.pt
This path is what `WanT2VOnlineEncoder` looks for by default.

Usage:
    python scripts/export_wan_prompt_embedding.py
    python scripts/export_wan_prompt_embedding.py --prompt "robot manipulation scene"
"""

from __future__ import annotations

import argparse
import os
import pathlib

import torch

DEFAULT_PROMPT = "a video of a scene"
DEFAULT_OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "openpi_vega3d" / "towers" / "wan_prompt_embedding.pt"
)
DEFAULT_TOKENIZER_DIR = "/workspace/openpi-Vega3D/ckpts/Wan2.1-T2V-1.3B/google/umt5-xxl"
HF_MODEL_ID = "google/umt5-xxl"
TEXT_LEN = 512  # WAN's expected prompt sequence length (T5_CONTEXT_TOKEN_NUMBER).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Text prompt to encode (default: %(default)r).")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT),
                        help="Where to save the embedding (default: location WAN tower looks for).")
    parser.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR,
                        help="Local umt5-xxl tokenizer dir (shipped with the WAN checkpoint).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[export] Embedding already exists at {out_path}; overwriting.")

    # Tokenizer: use the local umt5-xxl assets shipped with the WAN checkpoint.
    # Falls back to HF download if the local dir is missing.
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer_src = args.tokenizer_dir if os.path.isdir(args.tokenizer_dir) else HF_MODEL_ID
    print(f"[export] Loading tokenizer from {tokenizer_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    print(f"[export] Loading T5 encoder weights from {HF_MODEL_ID} (~11 GB on first run) ...")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    encoder = T5EncoderModel.from_pretrained(HF_MODEL_ID, torch_dtype=dtype)
    encoder.eval().requires_grad_(False)
    encoder.to(args.device)

    print(f"[export] Encoding prompt: {args.prompt!r}")
    tokens = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TEXT_LEN,
    ).to(args.device)

    with torch.no_grad():
        out = encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
    hidden = out.last_hidden_state.squeeze(0).detach().cpu().contiguous()  # [L, D]

    print(f"[export] Embedding shape: {tuple(hidden.shape)}, dtype: {hidden.dtype}")

    torch.save(hidden, out_path)
    print(f"[export] Saved to {out_path}")


if __name__ == "__main__":
    main()
