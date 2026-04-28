"""Diagnostic script to verify the openpi-Vega3D environment is correctly set up.

Checks all dependencies, paths, and assets needed for BEHAVIOR rollouts.

Usage:
    python scripts/verify_env.py
    # Or from the rollout launcher (inherits all env vars):
    bash scripts/run_rollout.sh  # then Ctrl-C and run this instead
"""

import importlib
import os
import sys
from pathlib import Path


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    if condition:
        results.append((PASS, name, detail))
    elif warn_only:
        results.append((WARN, name, detail))
    else:
        results.append((FAIL, name, detail))


def try_import(module_name: str, warn_only: bool = False) -> bool:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
        check(f"import {module_name}", True, f"version={version}")
        return True
    except Exception as e:
        check(f"import {module_name}", False, str(e), warn_only=warn_only)
        return False


def main():
    print("=" * 60)
    print("  openpi-Vega3D Environment Verification")
    print("=" * 60)
    print()

    # ── Python version ────────────────────────────────────────────────────
    py_ver = sys.version_info
    check(
        "Python version",
        py_ver.major == 3 and py_ver.minor == 10,
        f"{py_ver.major}.{py_ver.minor}.{py_ver.micro} (expect 3.10.x)",
    )

    # ── Core ML dependencies ──────────────────────────────────────────────
    torch_ok = try_import("torch")
    if torch_ok:
        import torch
        check(
            "torch CUDA",
            torch.cuda.is_available(),
            f"device_count={torch.cuda.device_count()}" if torch.cuda.is_available() else "no CUDA",
            warn_only=True,
        )

    try_import("jax")
    try_import("flax")
    try_import("transformers")
    try_import("safetensors")

    # ── openpi ────────────────────────────────────────────────────────────
    try_import("openpi")
    try_import("openpi.models_pytorch.pi0_pytorch")
    try_import("openpi.policies.b1k_policy")

    # ── openpi_vega3d ─────────────────────────────────────────────────────
    try_import("openpi_vega3d")
    try_import("openpi_vega3d.env")
    try_import("openpi_vega3d.policy_utils")
    try_import("openpi_vega3d.towers")

    # ── Isaac Sim Python package (ships with RLinf venv, not openpi-Vega3D) ─
    rlinf_sp = os.environ.get(
        "RLINF_SITE_PACKAGES",
        "/workspace/RLinf/.venv/lib/python3.10/site-packages",
    )
    # Append so RLinf does not shadow this venv's transformers / torch / etc.
    if os.path.isdir(rlinf_sp) and rlinf_sp not in sys.path:
        sys.path.append(rlinf_sp)
    try_import("isaacsim", warn_only=True)

    # ── OmniGibson / BDDL (may fail without ISAAC_PATH) ──────────────────
    try_import("omnigibson", warn_only=True)
    try_import("bddl", warn_only=True)

    # ── Optional: towers ──────────────────────────────────────────────────
    try_import("diffusers", warn_only=True)
    try_import("easydict", warn_only=True)
    try_import("einops", warn_only=True)

    # ── Optional: gello ───────────────────────────────────────────────────
    try_import("gello", warn_only=True)

    # ── Environment variables ─────────────────────────────────────────────
    isaac_path = os.environ.get("ISAAC_PATH", "")
    check(
        "ISAAC_PATH set",
        bool(isaac_path),
        isaac_path or "not set",
        warn_only=True,
    )
    if isaac_path:
        check(
            "ISAAC_PATH exists",
            os.path.isdir(isaac_path),
            isaac_path,
            warn_only=True,
        )

    og_data = os.environ.get("OMNIGIBSON_DATA_PATH", "")
    check(
        "OMNIGIBSON_DATA_PATH set",
        bool(og_data),
        og_data or "not set",
        warn_only=True,
    )
    if og_data:
        check(
            "OMNIGIBSON_DATA_PATH exists",
            os.path.isdir(og_data),
            og_data,
            warn_only=True,
        )
        check(
            "behavior-1k-assets present",
            os.path.isdir(os.path.join(og_data, "behavior-1k-assets")),
            "",
            warn_only=True,
        )
        check(
            "omnigibson-robot-assets present",
            os.path.isdir(os.path.join(og_data, "omnigibson-robot-assets")),
            "",
            warn_only=True,
        )

    carb = os.environ.get("CARB_APP_PATH", "")
    check("CARB_APP_PATH set", bool(carb), carb or "not set", warn_only=True)

    exp = os.environ.get("EXP_PATH", "")
    check("EXP_PATH set", bool(exp), exp or "not set", warn_only=True)

    # ── Filesystem assets ─────────────────────────────────────────────────
    default_ckpt = "/workspace/RLinf/safetensors_ckpts/openpi_05_20251115_050323_9000_tor/model.safetensors"
    check(
        "B1K checkpoint",
        os.path.isfile(default_ckpt),
        default_ckpt,
        warn_only=True,
    )

    norm_stats_candidates = [
        "/workspace/BEHAVIOR-1K/outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos/norm_stats.json",
        "/workspace/openpi/outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos/norm_stats.json",
    ]
    found_norm = any(os.path.isfile(p) for p in norm_stats_candidates)
    check(
        "norm_stats.json",
        found_norm,
        next((p for p in norm_stats_candidates if os.path.isfile(p)), "not found"),
        warn_only=True,
    )

    tokenizer = Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    check(
        "PaliGemma tokenizer",
        tokenizer.is_file(),
        str(tokenizer),
        warn_only=True,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print()
    passes = sum(1 for s, _, _ in results if s == PASS)
    warnings = sum(1 for s, _, _ in results if s == WARN)
    failures = sum(1 for s, _, _ in results if s == FAIL)

    for status, name, detail in results:
        tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
        line = f"  {tag:6s} {name}"
        if detail:
            line += f"  ({detail})"
        print(line)

    print()
    print(f"  Total: {len(results)} checks  |  {passes} passed  |  {warnings} warnings  |  {failures} failed")
    print()

    if failures > 0:
        print("  Some required checks failed. Review output above.")
        sys.exit(1)
    elif warnings > 0:
        print("  All required checks passed. Some optional features are unavailable (see WARN above).")
    else:
        print("  All checks passed!")


if __name__ == "__main__":
    main()
