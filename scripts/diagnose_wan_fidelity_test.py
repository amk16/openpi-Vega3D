"""CI-safe smoke test for scripts/diagnose_wan_fidelity.py (Phase 8.5).

Builds a tiny synthetic feature cache with a planted pillarbox signature and
runs the column-energy diagnostic against it — no remote, no checkpoint, no
real cache needed (acceptance criterion 7 of docs/PHASE8_PLAN.md).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import pytest


def _load_diagnose_module():
    path = pathlib.Path(__file__).parent / "diagnose_wan_fidelity.py"
    spec = importlib.util.spec_from_file_location("diagnose_wan_fidelity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_column_energy_synthetic_cache(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    grid, feat_dim, frames = 16, 8, 3
    cam_dir = tmp_path / "base_0_rgb"
    cam_dir.mkdir()

    # Plant the Break-1 signature: ~20x weaker energy in the 3 outermost
    # columns each side (pure black-bar tokens), saved in the real cache
    # format ([T, S*S, C] bfloat16 under the "features" key).
    g = torch.randn(frames, grid, grid, feat_dim)
    g[:, :, :3, :] *= 0.05
    g[:, :, grid - 3 :, :] *= 0.05
    feats = g.reshape(frames, grid * grid, feat_dim).to(torch.bfloat16)
    for ep in range(2):
        safetensors_torch.save_file({"features": feats}, str(cam_dir / f"ep_{ep:06d}.safetensors"))
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "tower_name": "wan_t2v",
                "output_spatial": grid,
                "window": 1,
                "stride": 1,
                "feat_block_idx": 20,
                "content_region_pool": False,
            }
        )
    )

    mod = _load_diagnose_module()
    args = argparse.Namespace(cache_dir=str(tmp_path), camera=None, episodes=5, grid=16)
    mod.run_column_energy(args)  # must not raise

    out = capsys.readouterr().out
    assert "edge/middle ratio" in out
    # The planted signature must be detected: edge/middle ratio well below 1.
    ratio_line = next(line for line in out.splitlines() if "edge/middle ratio" in line)
    ratio = float(ratio_line.split(":")[1])
    assert ratio < 0.3, f"planted pillarbox signature not detected (ratio={ratio})"
