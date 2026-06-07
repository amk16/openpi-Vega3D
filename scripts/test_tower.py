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

# ruff: noqa: PLC0415 — heavy imports (torch, towers) are deferred into the
# functions so --offline mode and pytest collection stay import-light.

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
    from openpi_vega3d.towers.diagnostics import log_tower_registry_keys
    from openpi_vega3d.towers.diagnostics import run_base_tower_contract_smoke

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


# Not a pytest test (CLI helper taking args); without this, pytest collection
# errors on the missing 'tower_name' fixture.
test_tower.__test__ = False


# ---------------------------------------------------------------------------
# Phase 8.5 — pytest tests for the Break-1 fix (content-region pooling).
# Test 1 is CI-safe: pure tensor math against letterbox_content_box() plus a
# standalone slice-and-pool, no WAN checkpoint. The full-encoder variant is
# behind `-m manual` for the remote.
# ---------------------------------------------------------------------------

try:
    import pytest

    _manual_mark = pytest.mark.manual
except ImportError:  # CLI use in an env without pytest

    def _manual_mark(f):
        return f


def test_pillarbox_content_pooling_geometry():
    """Phase 8.5 test 1: content slice excludes every pad token; pooled
    quadrant geometry maps correctly to the 16x16 grid."""
    import torch
    import torch.nn.functional as F  # noqa: N812 — torch convention

    from openpi_vega3d.towers.common import letterbox_content_box

    # Exact box for the LIBERO headline case (224^2 -> 832x480, 16 px/token):
    # pillarbox is exactly 11 tokens per side -> 30x30 content at cols 11:41.
    assert letterbox_content_box(224, 224, 480, 832, 16) == (0, 30, 11, 41)
    # Input aspect == canvas aspect -> no pad, full grid.
    assert letterbox_content_box(480, 832, 480, 832, 16) == (0, 30, 0, 52)

    # Standalone slice-and-pool, mirroring the encoder's
    # _slice_content_region + adaptive_avg_pool2d (without a checkpoint).
    pad_value = -7.0
    grid = torch.full((1, 4, 30, 52), pad_value)  # [B, C, grid_h, grid_w]
    top, bottom, left, right = letterbox_content_box(224, 224, 480, 832, 16)
    grid[..., top:bottom, left:right] = 1.0
    # Bright top-left quadrant of the CONTENT region (rows 0:15, content cols 11:26).
    grid[..., 0:15, 11:26] = 10.0

    # Legacy behavior (the break): pooling the full canvas leaks pad into the
    # outer output columns.
    legacy = F.adaptive_avg_pool2d(grid, (16, 16))
    assert legacy.min() < 0, "sanity: legacy full-canvas pooling must be pad-contaminated"

    # Fixed behavior: slice first -> zero pad contribution to any pooled token.
    sliced = grid[..., top:bottom, left:right]
    assert sliced.shape[-2:] == (30, 30)
    assert (sliced != pad_value).all(), "content slice must exclude every pad token"
    fixed = F.adaptive_avg_pool2d(sliced, (16, 16))
    assert fixed.min() > 0, "no pad value may reach any pooled token"

    # Quadrant geometry: bright top-left half of the content must land in the
    # top-left 8x8 of the 16x16 output, and only there.
    top_left = fixed[..., :8, :8].mean()
    bottom_right = fixed[..., 8:, 8:].mean()
    assert top_left > 8.0, f"bright quadrant should pool to ~10, got {top_left}"
    assert bottom_right < 2.0, f"dark quadrant should pool to ~1, got {bottom_right}"

    # No output column may be systematically pad-dominated.
    assert (fixed.amin(dim=(0, 1, 2)) > 0).all(), "every output column must be pad-free"


@_manual_mark
def test_pillarbox_content_pooling_full_encoder():
    """Phase 8.5 test 1, full-encoder variant (remote; requires WAN checkpoint).

    Run: WAN_T2V_CKPT_DIR=... uv run pytest scripts/test_tower.py -m manual
    """
    import os

    import pytest
    import torch

    ckpt = os.environ.get("WAN_T2V_CKPT_DIR")
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip("WAN_T2V_CKPT_DIR not set / not a directory (remote-only test)")

    from openpi_vega3d.towers import TOWER_REGISTRY

    tower = TOWER_REGISTRY["wan_t2v"](checkpoint_dir=ckpt, output_spatial=16, content_region_pool=True)
    if torch.cuda.is_available():
        tower = tower.to("cuda")
    images = torch.rand(1, 3, 224, 224).to(next(tower.parameters()).device)
    feats = tower.encode(images)
    assert feats.shape[1] == 256, f"expected 256 tokens, got {feats.shape}"
    # Column-energy flatness: with the content slice active there are no pure
    # black-bar columns, so no column should be an extreme outlier.
    g = feats.float().reshape(1, 16, 16, -1).norm(dim=-1)  # [1, H, W]
    col = g.mean(dim=(0, 1))
    edge = torch.cat([col[:3], col[-3:]]).mean()
    middle = col[5:11].mean()
    ratio = (edge / middle).item()
    assert 0.5 < ratio < 2.0, f"edge/middle column energy ratio {ratio:.3f} suggests pad tokens leaked"


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
