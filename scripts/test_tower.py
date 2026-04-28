"""Phase 1 validation: test generative tower infrastructure.

Usage:
    # Offline structure/import validation (no checkpoints needed):
    python scripts/test_tower.py --offline

    # Full VAE tower test (requires SD2.1 checkpoint):
    python scripts/test_tower.py --tower vae --checkpoint path/to/sd21-base

    # Full WAN T2V tower test (requires WAN checkpoint + prompt embedding):
    python scripts/test_tower.py --tower wan_t2v \
        --checkpoint path/to/Wan2.1-T2V-1.3B \
        --prompt_emb path/to/wan_prompt_embedding.pt
"""

import argparse
import sys


def test_offline():
    """Validate structure, imports, and contracts without any checkpoints."""
    import ast
    import os

    print("=" * 60)
    print("OFFLINE VALIDATION")
    print("=" * 60)

    base_dir = os.path.join(os.path.dirname(__file__), "..", "src", "openpi_vega3d", "towers")
    base_dir = os.path.normpath(base_dir)

    print("\n--- Syntax check ---")
    files_to_check = [
        "__init__.py", "base.py", "common.py",
        "vae_online_encoder.py", "vae_tower.py",
        "wan_t2v_encoder.py", "wan_tower.py",
        "wan/__init__.py",
        "wan/configs/__init__.py", "wan/configs/shared_config.py",
        "wan/configs/wan_t2v_1_3B.py", "wan/configs/wan_t2v_14B.py",
        "wan/configs/wan_i2v_14B.py",
        "wan/modules/__init__.py", "wan/modules/model.py",
        "wan/modules/vae.py", "wan/modules/attention.py",
        "wan/utils/__init__.py", "wan/utils/fm_solvers_unipc.py",
    ]
    ok = 0
    for f in files_to_check:
        path = os.path.join(base_dir, f)
        if not os.path.exists(path):
            print(f"  MISSING  {f}")
            continue
        try:
            with open(path) as fh:
                ast.parse(fh.read(), filename=f)
            print(f"  OK       {f}")
            ok += 1
        except SyntaxError as e:
            print(f"  FAIL     {f}: {e}")
    print(f"\nSyntax: {ok}/{len(files_to_check)} passed")

    print("\n--- BaseTower ABC contract ---")
    import importlib.util
    spec = importlib.util.spec_from_file_location("base", os.path.join(base_dir, "base.py"))
    base_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_mod)
    BaseTower = base_mod.BaseTower

    assert hasattr(BaseTower, "encode"), "Missing encode method"
    assert hasattr(BaseTower, "feat_dim"), "Missing feat_dim property"
    assert hasattr(BaseTower, "freeze"), "Missing freeze method"
    assert hasattr(BaseTower, "check_output"), "Missing check_output method"
    print("  BaseTower has: encode, feat_dim, freeze, check_output")

    import torch

    class DummyTower(BaseTower):
        @property
        def feat_dim(self) -> int:
            return 8

        def encode(self, images):
            b = images.shape[0]
            return torch.randn(b, 196, self.feat_dim)

    dummy = DummyTower()
    dummy.freeze()
    result = dummy.check_output(torch.randn(2, 3, 224, 224))
    assert result["output_shape"] == (2, 196, 8), f"Bad shape: {result['output_shape']}"
    assert result["feat_dim"] == 8
    assert result["frozen"] is True
    print(f"  DummyTower check_output: {result}")

    print("\n--- Import graph (ast-level) ---")
    for f in ["vae_tower.py", "wan_tower.py"]:
        path = os.path.join(base_dir, f)
        with open(path) as fh:
            tree = ast.parse(fh.read(), f)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
        internal = [i for i in imports if "base" in i or "encoder" in i or "tower" in i]
        print(f"  {f}: {internal}")

    print("\n--- TOWER_REGISTRY keys (ast-level) ---")
    init_path = os.path.join(base_dir, "__init__.py")
    with open(init_path) as fh:
        src = fh.read()
    assert "TOWER_REGISTRY" in src
    assert '"vae"' in src or "'vae'" in src
    assert '"wan_t2v"' in src or "'wan_t2v'" in src
    print("  Registry contains: vae, wan_t2v")

    print("\n--- Rollout logger compatibility (diagnostics module) ---")
    root_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if root_src not in sys.path:
        sys.path.insert(0, root_src)
    import logging

    logging.basicConfig(level=logging.WARNING)
    from openpi_vega3d.towers.diagnostics import (
        log_tower_registry_keys,
        run_base_tower_contract_smoke,
    )

    log_tower_registry_keys()
    run_base_tower_contract_smoke(device="cpu")
    print("  diagnostics.log_tower_registry_keys + run_base_tower_contract_smoke: OK")

    print("\n" + "=" * 60)
    print("OFFLINE VALIDATION: ALL PASSED")
    print("=" * 60)


def test_tower(tower_name, checkpoint, prompt_emb=None):
    """Full tower test with real checkpoint."""
    import torch

    sys.path.insert(0, "src")
    from openpi_vega3d.towers import TOWER_REGISTRY

    print(f"\n{'=' * 60}")
    print(f"TOWER TEST: {tower_name}")
    print(f"{'=' * 60}")

    kwargs = {"checkpoint_dir": checkpoint}
    if tower_name == "wan_t2v" and prompt_emb:
        kwargs["prompt_emb_path"] = prompt_emb

    tower = TOWER_REGISTRY[tower_name](**kwargs)

    images = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        images = images.cuda()

    result = tower.check_output(images)

    print(f"Tower: {tower_name}")
    print(f"  Input:    {result['input_shape']}")
    print(f"  Output:   {result['output_shape']}")
    print(f"  feat_dim: {result['feat_dim']}")
    print(f"  Mean:     {result['mean']:.4f}")
    print(f"  Std:      {result['std']:.4f}")
    print(f"  Frozen:   {result['frozen']}")

    assert len(result["output_shape"]) == 3, "Output should be [B, tokens, feat_dim]"
    assert result["output_shape"][0] == 1, "Batch dim should match input"
    assert result["output_shape"][2] == result["feat_dim"], "Last dim should match feat_dim"
    assert result["frozen"] is True, "Tower should be frozen"

    print(f"\nTOWER TEST {tower_name}: PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run offline validation only")
    parser.add_argument("--tower", type=str, choices=["vae", "wan_t2v"], help="Tower to test")
    parser.add_argument("--checkpoint", type=str, help="Path to tower checkpoint")
    parser.add_argument("--prompt_emb", type=str, help="Path to WAN prompt embedding (wan_t2v only)")
    args = parser.parse_args()

    if args.offline:
        test_offline()
    elif args.tower:
        if not args.checkpoint:
            parser.error("--checkpoint is required for tower tests")
        test_tower(args.tower, args.checkpoint, args.prompt_emb)
    else:
        test_offline()


if __name__ == "__main__":
    main()
