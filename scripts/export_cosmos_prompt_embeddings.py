"""Precompute T5 prompt embeddings for Cosmos tower text conditioning.

Cosmos-Predict2.5-2B uses T5-11B (google-t5/t5-11b) as its text encoder.
Rather than loading ~22GB of T5 weights during tower feature precomputation
or training, this script encodes all unique task prompts once and saves
them as a lookup table.

The output cache is consumed by PromptEmbeddingCache
(openpi_vega3d.towers.prompt_cache).

Usage:
    # Encode all prompts from the LIBERO dataset:
    python scripts/export_cosmos_prompt_embeddings.py --repo_id physical-intelligence/libero

    # Encode a specific list of prompts:
    python scripts/export_cosmos_prompt_embeddings.py \
        --prompts "pick up the cup" "open the drawer"

    # Custom output path:
    python scripts/export_cosmos_prompt_embeddings.py --repo_id physical-intelligence/libero \
        --out_path /workspace/my_cache/cosmos_prompt_embeddings.pt
"""

from __future__ import annotations

import argparse
import pathlib

import torch

DEFAULT_OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src" / "openpi_vega3d" / "towers" / "cosmos_prompt_embeddings.pt"
)
HF_T5_MODEL = "google-t5/t5-11b"
MAX_SEQ_LEN = 512


def get_prompts_from_dataset(repo_id: str) -> list[str]:
    """Extract all unique task prompts from a LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print(f"[export] Loading dataset {repo_id} ...")
    ds = LeRobotDataset(repo_id)
    if hasattr(ds, "meta") and hasattr(ds.meta, "tasks") and ds.meta.tasks:
        prompts = sorted(set(ds.meta.tasks.values()))
    elif "task" in ds.hf_dataset.column_names:
        prompts = sorted(set(ds.hf_dataset["task"]))
    else:
        raise ValueError(
            f"Cannot extract prompts from {repo_id}: no 'task' column or meta.tasks. "
            "Pass --prompts explicitly instead."
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo_id", type=str, help="LeRobot dataset repo ID to extract prompts from.")
    group.add_argument("--prompts", nargs="+", type=str, help="Explicit list of prompts to encode.")
    group.add_argument("--prompts_file", type=str, help="Path to a text file with one prompt per line.")
    parser.add_argument("--out_path", default=str(DEFAULT_OUT), help="Output cache path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        prompts = [
            line.strip()
            for line in pathlib.Path(args.prompts_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        prompts = get_prompts_from_dataset(args.repo_id)
    print(f"[export] {len(prompts)} unique prompts to encode")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p!r}")

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, T5EncoderModel

    print(f"[export] Loading T5 tokenizer and encoder from {HF_T5_MODEL} (~22GB on first run) ...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    pt_dtype = dtype_map[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(HF_T5_MODEL)
    encoder = T5EncoderModel.from_pretrained(HF_T5_MODEL, torch_dtype=pt_dtype)
    encoder.eval().requires_grad_(False)
    encoder.to(args.device)

    embeddings: dict[str, torch.Tensor] = {}
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
        ).to(args.device)

        with torch.no_grad():
            out = encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
        embed = out.last_hidden_state.squeeze(0)  # [seq_len, embed_dim]

        # Zero-fill padding positions (matches Cosmos pipeline convention).
        mask = tokens["attention_mask"].squeeze(0).bool()
        embed[~mask] = 0

        embeddings[prompt] = embed.detach().cpu().contiguous()

    embed_dim = next(iter(embeddings.values())).shape[-1]
    print(f"[export] embed_dim={embed_dim}, seq_len={MAX_SEQ_LEN}, dtype={pt_dtype}")

    cache = {
        "embeddings": embeddings,
        "embed_dim": embed_dim,
        "seq_len": MAX_SEQ_LEN,
        "t5_model": HF_T5_MODEL,
    }
    torch.save(cache, out_path)
    print(f"[export] Saved {len(embeddings)} prompt embeddings to {out_path}")


if __name__ == "__main__":
    main()
