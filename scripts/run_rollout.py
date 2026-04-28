"""Minimal BEHAVIOR rollout script for openpi-vega3d.

Runs a Pi05 policy in the OmniGibson BEHAVIOR environment using receding
horizon control.  No RLinf dependency -- uses SimpleEnv and load_b1k_policy
directly.

**Alignment with RLinf ``minimal_rollout.py`` (setup order)**

1. **Phase 0 — Import priming** (deviation from minimal_rollout): import the
   OpenPI / PyTorch / Transformers graph *before* Isaac Sim starts. Kit adds
   ``omni.isaac.ml_archive/.../pip_prebundle/torch`` to ``sys.path``; if that
   shadows ``torch.nn.attention`` while ``torch._higher_order_ops`` still comes
   from the venv, imports fail (e.g. ``TransformGetItemToIndex`` mismatch) and
   teardown can segfault.
2. **Phase 1 — Environment**: build OmniGibson config (default: same
   ``r1pro_behavior.yaml`` + overrides as ``create_env_config`` in
   ``minimal_rollout.py``), then construct ``SimpleEnv`` / ``VectorEnvironment``,
   then optional ``load_task_instance``.
3. **Phase 2 — Policy**: ``load_b1k_policy`` (openpi ``Policy``, not RLinf's
   OpenPI action model — we are not running PPO).
4. **Phase 3 — Rollout**: ``reset()`` + receding-horizon loop with
   ``policy.infer()`` (same *role* as ``predict_action_batch`` in minimal
   rollout, different API).

**Intentional differences from minimal_rollout**

- No RL: no buffer, GAE, PPO, value head, or flow log-prob plumbing.
- Policy stack: Hugging Face ``Policy.infer`` + B1K transforms, not
  ``rlinf``'s ``predict_action_batch``.
- You may still use ``--env_setup gello`` for the older gello-generated config
  (task-specific scene) if you need parity with ``SimpleEnv``'s default before
  this change.

Logging (similar spirit to RLinf ``minimal_rollout.py`` diagnostics):
    - Console + optional file (default: ``outputs/rollouts/run_rollout_<task>_<timestamp>.log``)
    - Phase-style messages: ``[phase]`` prefix for policy / env / rollout
    - Optional JSON summary at episode end (``--json_out`` or alongside log file)

Usage:
    python scripts/run_rollout.py \
        --task_name turning_on_radio \
        --instance_id 1 \
        --ckpt_dir outputs/checkpoints/openpi_05_.../9000 \
        --norm_stats_dir outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos

Flags:
    --use_vega3d        Enable VEGA-3D spatial features (Phase 4)
    --chunk_size        Actions executed from each prediction before re-querying (default: 100)
    --max_steps         Max environment steps per episode (default: from env)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("run_rollout")


def configure_logging(
    *,
    level: int,
    log_file: str | None,
    quiet_third_party: bool = True,
) -> str | None:
    """Attach console + optional file handlers to the root logger. Returns log file path or None."""
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    path_written: str | None = None
    fh: logging.FileHandler | None = None
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        path_written = str(p.resolve())

    # OmniGibson/Isaac Sim raises the root logger to WARNING after Kit starts, which
    # would drop our INFO phase/trace lines. Keep this script's logger independent.
    rr = logging.getLogger("run_rollout")
    rr.handlers.clear()
    rr.setLevel(level)
    rr.propagate = False
    rr.addHandler(sh)
    if fh is not None:
        rr.addHandler(fh)

    if quiet_third_party and level > logging.DEBUG:
        for name in ("matplotlib", "PIL", "urllib3", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

    return path_written


def log_phase(phase: str, msg: str, *args, **kwargs) -> None:
    """RLinf-style phase tag so grepping logs is easy."""
    _LOG.info("[%s] " + msg, phase, *args, **kwargs)


def _flush_all_log_handlers() -> None:
    """Flush file + console handlers so the last line survives abrupt exit (e.g. segfault)."""
    for h in logging.root.handlers:
        flush = getattr(h, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
    sys.stdout.flush()
    sys.stderr.flush()


def log_trace(step: str, msg: str, *args, **kwargs) -> None:
    """Monotonic fine-grained marker; always flushes handlers."""
    _LOG.info("[trace %s] " + msg, step, *args, **kwargs)
    _flush_all_log_handlers()


def _summarize_obs_for_log(obs: dict) -> dict:
    """Small JSON-serializable summary of observation tensors."""
    out: dict = {}
    for k, v in obs.items():
        if k in ("task_description",):
            out[k] = (str(v)[:120] + "…") if len(str(v)) > 120 else str(v)
        elif hasattr(v, "shape"):
            arr = np.asarray(v)
            out[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }
        else:
            out[k] = v
    return out


def _run_basename(task_name: str, ts: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)[:80]
    return f"run_rollout_{safe}_{ts}"


def build_omnigibson_cfg_r1pro_like_minimal_rollout(
    task_name: str,
    max_episode_steps: int,
) -> dict:
    """Mirror ``minimal_rollout.create_env_config`` omni section: ``r1pro_behavior.yaml`` + OpenPI overrides.

    See RLinf ``minimal_rollout.py`` ``create_env_config`` (lines ~193--206).
    """
    import omnigibson as og
    from omegaconf import OmegaConf
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    omnigibson_cfg = OmegaConf.load(config_filename)
    omnigibson_cfg["task"]["termination_config"]["max_steps"] = max_episode_steps
    omnigibson_cfg["robots"][0]["name"] = "robot_r1"
    omnigibson_cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    omnigibson_cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    omnigibson_cfg["task"]["activity_name"] = task_name
    omnigibson_cfg["task"]["include_obs"] = False
    return OmegaConf.to_container(omnigibson_cfg, resolve=True)


def parse_args():
    p = argparse.ArgumentParser(description="openpi-vega3d BEHAVIOR rollout")

    p.add_argument(
        "--log_level",
        type=str,
        default=os.environ.get("ROLLOUT_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level (default: INFO or ROLLOUT_LOG_LEVEL)",
    )
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Write full log to this path. Default: outputs/rollouts/run_rollout_<task>_<utc_time>.log",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for default log file (default: $ROLLOUT_LOG_DIR or <repo>/outputs/rollouts)",
    )
    p.add_argument(
        "--no_log_file",
        action="store_true",
        help="Console only; do not write a .log file",
    )
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Write episode summary JSON here. Default: same basename as --log_file with .json",
    )
    p.add_argument(
        "--no_json_out",
        action="store_true",
        help="Do not write episode summary JSON",
    )

    p.add_argument("--task_name", type=str, required=True, help="BEHAVIOR task name")
    p.add_argument("--task_description", type=str, default=None,
                    help="Natural language task description (auto-loaded if omitted)")
    p.add_argument("--instance_id", type=int, default=0, help="TRO instance ID")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Model checkpoint directory")
    p.add_argument("--norm_stats_dir", type=str,
                    default="outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos",
                    help="Norm stats directory")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    p.add_argument("--max_steps", type=int, default=None, help="Max steps (default: env max)")
    p.add_argument("--chunk_size", type=int, default=100,
                    help="Number of actions executed from each policy prediction before re-querying the model")
    p.add_argument("--action_horizon", type=int, default=100,
                    help="Action chunk size predicted by the model")

    p.add_argument("--use_vega3d", action="store_true", default=False,
                    help="Enable VEGA-3D adaptive gated fusion in the policy (Phase 3). "
                         "Uses --gen_tower / --gen_tower_ckpt / --gen_tower_prompt_emb for tower config.")
    p.add_argument("--force_gate", type=float, default=None,
                    help="VEGA-3D fusion gate ablation: force gate to this value in [0,1]. "
                         "0.0=pure generative, 1.0=pure semantic, None=learned (default).")

    p.add_argument(
        "--tower_arch_test",
        action="store_true",
        help=(
            "Log TOWER_REGISTRY keys and run a CPU BaseTower contract smoke (no checkpoint). "
            "Emits [trace tower] lines on the run_rollout logger."
        ),
    )
    p.add_argument(
        "--gen_tower",
        type=str,
        default=None,
        choices=("vae", "wan_t2v"),
        help="After env.reset(), load this tower and run one encode on head_rgb (needs --gen_tower_ckpt).",
    )
    p.add_argument(
        "--gen_tower_ckpt",
        type=str,
        default=None,
        help="Checkpoint directory for --gen_tower (SD2.1 base for vae, WAN root for wan_t2v).",
    )
    p.add_argument(
        "--gen_tower_prompt_emb",
        type=str,
        default=None,
        help="Path to WAN prompt embedding .pt (required for --gen_tower wan_t2v).",
    )

    p.add_argument(
        "--env_setup",
        type=str,
        choices=("r1pro_yaml", "gello"),
        default="r1pro_yaml",
        help=(
            "How to build OmniGibson config: "
            "'r1pro_yaml' = same as RLinf minimal_rollout (r1pro_behavior.yaml + proprio/rgb); "
            "'gello' = task-specific generate_basic_environment_config (requires joylo)."
        ),
    )
    p.add_argument("--og_cfg_path", type=str, default=None,
                    help="If set, load this YAML as omnigibson cfg (overrides --env_setup)")
    p.add_argument("--partial_scene_load", action="store_true", default=False,
                    help="Only for --env_setup gello: partial scene load (requires joylo)")
    p.add_argument("--disable_transition_rules", action="store_true", default=False)

    p.add_argument("--task_descriptions_path", type=str, default=None,
                    help="Path to task descriptions JSONL file")

    p.add_argument(
        "--skip_load_task_instance",
        action="store_true",
        default=False,
        help="Skip env.load_task_instance() (default TRO instance). For debugging only.",
    )

    return p.parse_args()


def load_task_description(task_name: str, descriptions_path: str | None = None) -> str:
    """Load the natural-language task description for a task."""
    if descriptions_path is not None:
        import json
        with open(descriptions_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("task_name") == task_name:
                    return entry["task"]
        _LOG.warning("Task %s not found in %s, using task name as prompt", task_name, descriptions_path)

    return task_name.replace("_", " ")


def format_obs_for_policy(obs: dict) -> dict:
    """Convert SimpleEnv observation into the dict format expected by Policy.infer().

    The B1kInputs transform expects:
        observation/egocentric_camera: (H, W, 3) uint8
        observation/wrist_image_left:  (H, W, 3) uint8
        observation/wrist_image_right: (H, W, 3) uint8
        observation/state:             (N,) float
        prompt:                        str
        task_index:                    int
    """
    return {
        "observation/egocentric_camera": obs["head_rgb"],
        "observation/wrist_image_left": obs["left_wrist_rgb"],
        "observation/wrist_image_right": obs["right_wrist_rgb"],
        "observation/state": obs["proprio"],
        "prompt": obs["task_description"],
        "task_index": obs["task_index"],
    }


def main() -> None:
    args = parse_args()
    if args.gen_tower and not args.gen_tower_ckpt:
        raise SystemExit("--gen_tower requires --gen_tower_ckpt")
    if args.gen_tower == "wan_t2v" and not args.gen_tower_prompt_emb:
        raise SystemExit("--gen_tower wan_t2v requires --gen_tower_prompt_emb")

    repo_root = Path(__file__).resolve().parent.parent
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _run_basename(args.task_name, ts)
    log_dir = Path(args.log_dir) if args.log_dir else Path(os.environ.get("ROLLOUT_LOG_DIR", repo_root / "outputs" / "rollouts"))

    log_file: str | None = None
    if not args.no_log_file:
        log_file = args.log_file or str(log_dir / f"{base}.log")

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=level, log_file=log_file, quiet_third_party=True)
    if log_file:
        _LOG.info("Log file: %s", log_file)

    task_desc = args.task_description or load_task_description(
        args.task_name, args.task_descriptions_path
    )

    json_out: str | None = None
    if not args.no_json_out:
        if args.json_out:
            json_out = args.json_out
        elif log_file:
            json_out = str(Path(log_file).with_suffix(".json"))
        else:
            json_out = str(log_dir / f"{base}.json")

    summary: dict = {
        "schema": "openpi_vega3d.run_rollout.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": args.task_name,
        "task_description": task_desc,
        "instance_id": args.instance_id,
        "ckpt_dir": args.ckpt_dir,
        "norm_stats_dir": args.norm_stats_dir,
        "device": args.device,
        "max_steps_requested": args.max_steps,
        "chunk_size": args.chunk_size,
        "action_horizon": args.action_horizon,
        "skip_load_task_instance": args.skip_load_task_instance,
        "log_file": log_file,
        "status": "running",
        "error": None,
    }

    _LOG.info("=" * 72)
    _LOG.info("openpi-Vega3D BEHAVIOR rollout")
    _LOG.info("=" * 72)
    _LOG.info("argv: %s", " ".join(sys.argv))
    _LOG.info("Task: %s | description: %s", args.task_name, task_desc[:200])
    log_phase("config", "ckpt_dir=%s", args.ckpt_dir)
    log_phase("config", "norm_stats_dir=%s", args.norm_stats_dir)
    log_phase("config", "device=%s chunk_size=%s action_horizon=%s", args.device, args.chunk_size, args.action_horizon)

    if args.tower_arch_test:
        from openpi_vega3d.towers.diagnostics import (
            log_tower_registry_keys,
            run_base_tower_contract_smoke,
        )

        log_phase(
            "tower",
            "TOWER_ARCH_TEST: registry + BaseTower contract smoke (CPU, no checkpoint)",
        )
        log_tower_registry_keys()
        run_base_tower_contract_smoke(device="cpu")

    # Episode length for OmniGibson task termination (matches minimal_rollout default 5000 when unset)
    max_episode_steps = args.max_steps if args.max_steps is not None else 5000
    summary["max_episode_steps"] = max_episode_steps
    summary["env_setup"] = args.env_setup

    try:
        # ------------------------------------------------------------------
        # PHASE 0: Prime venv torch + OpenPI before Kit mutates sys.path
        # ------------------------------------------------------------------
        log_phase(
            "phase0",
            "PHASE 0: import openpi stack before Isaac Sim (avoid ml_archive torch shadowing)",
        )
        log_trace("00", "before: from openpi_vega3d.policy_utils import load_b1k_policy")
        from openpi_vega3d.policy_utils import load_b1k_policy

        log_trace("01", "after: load_b1k_policy import OK (graph primed)")

        # ------------------------------------------------------------------
        # PHASE 1: Environment — same order as RLinf minimal_rollout.main()
        # ------------------------------------------------------------------
        log_trace("02", "before: from openpi_vega3d.env import SimpleEnv")
        from openpi_vega3d.env import SimpleEnv

        log_trace("03", "after: SimpleEnv module import OK")

        log_phase("phase1", "PHASE 1: ENVIRONMENT (Isaac Sim / OmniGibson)")
        t_env0 = time.perf_counter()

        og_cfg: dict | None = None
        log_trace("04", "building og_cfg (env_setup=%s)", args.env_setup)
        if args.og_cfg_path is not None:
            from omegaconf import OmegaConf
            og_cfg = OmegaConf.to_container(OmegaConf.load(args.og_cfg_path), resolve=True)
            log_phase("env", "Using omnigibson cfg from %s (--og_cfg_path overrides --env_setup)", args.og_cfg_path)
        elif args.env_setup == "r1pro_yaml":
            og_cfg = build_omnigibson_cfg_r1pro_like_minimal_rollout(
                args.task_name, max_episode_steps=max_episode_steps
            )
            log_phase(
                "env",
                "Built omni cfg from r1pro_behavior.yaml (minimal_rollout.create_env_config pattern), max_episode_steps=%d",
                max_episode_steps,
            )
        else:
            log_phase(
                "env",
                "Using gello task config (--env_setup gello); partial_scene_load=%s",
                args.partial_scene_load,
            )

        log_trace("05", "og_cfg ready; about to construct SimpleEnv(...)")
        env = SimpleEnv(
            task_name=args.task_name,
            task_description=task_desc,
            omnigibson_cfg=og_cfg,
            partial_scene_load=args.partial_scene_load,
            disable_transition_rules=args.disable_transition_rules,
        )
        log_trace("06", "SimpleEnv constructor returned OK")
        log_phase("env", "VectorEnvironment ready in %.1fs", time.perf_counter() - t_env0)

        if args.skip_load_task_instance:
            log_phase("task", "Skipping load_task_instance (--skip_load_task_instance)")
            log_trace("07", "skipped load_task_instance")
        else:
            log_trace("07a", "about to env.load_task_instance(%s)", args.instance_id)
            log_phase("task", "Loading TRO instance id=%s ...", args.instance_id)
            env.load_task_instance(args.instance_id)
            log_trace("07b", "env.load_task_instance returned OK")
            log_phase("task", "Instance load complete")

        # ------------------------------------------------------------------
        # PHASE 2: Policy — openpi load only (no RLinf model wrapper / PPO)
        # ------------------------------------------------------------------
        log_phase("phase2", "PHASE 2: LOAD POLICY (openpi Policy.infer — not RLinf predict_action_batch)")
        t_policy0 = time.perf_counter()
        log_phase("policy", "Loading policy from %s", args.ckpt_dir)
        log_trace("08", "about to load_b1k_policy(ckpt=%s, device=%s)", args.ckpt_dir, args.device)

        # Build VEGA-3D tower kwargs from the --gen_tower* flags if fusion is requested.
        vega3d_tower_kwargs = None
        if args.use_vega3d:
            if not args.gen_tower or not args.gen_tower_ckpt:
                raise ValueError(
                    "--use_vega3d requires --gen_tower and --gen_tower_ckpt to be set"
                )
            vega3d_tower_kwargs = {"checkpoint_dir": args.gen_tower_ckpt}
            if args.gen_tower == "wan_t2v":
                if not args.gen_tower_prompt_emb:
                    raise ValueError(
                        "--gen_tower wan_t2v requires --gen_tower_prompt_emb"
                    )
                vega3d_tower_kwargs["prompt_emb_path"] = args.gen_tower_prompt_emb
            log_phase(
                "policy",
                "VEGA-3D fusion enabled: tower=%s ckpt=%s force_gate=%s",
                args.gen_tower, args.gen_tower_ckpt, args.force_gate,
            )

        policy = load_b1k_policy(
            checkpoint_dir=args.ckpt_dir,
            norm_stats_dir=args.norm_stats_dir,
            device=args.device,
            action_horizon=args.action_horizon,
            use_vega3d=args.use_vega3d,
            vega3d_tower_name=args.gen_tower if args.use_vega3d else "vae",
            vega3d_tower_kwargs=vega3d_tower_kwargs,
            vega3d_force_gate=args.force_gate,
        )
        log_trace("09", "load_b1k_policy returned OK")
        log_phase("policy", "Loaded in %.1fs", time.perf_counter() - t_policy0)

        # ------------------------------------------------------------------
        # PHASE 3: Rollout — receding horizon (same role as minimal_rollout loop)
        # ------------------------------------------------------------------
        max_steps = args.max_steps or env.max_steps
        summary["max_steps_effective"] = max_steps
        log_phase("phase3", "PHASE 3: ROLLOUT max_steps=%d chunk_size=%d", max_steps, args.chunk_size)

        t_roll0 = time.perf_counter()
        log_phase("rollout", "Calling env.reset() ...")
        log_trace("10", "about to env.reset()")
        obs = env.reset()
        log_trace("11", "env.reset() returned OK")
        _LOG.debug("First obs summary: %s", json.dumps(_summarize_obs_for_log(obs), indent=2))
        log_phase("rollout", "reset() done; keys=%s", list(obs.keys()))

        if args.gen_tower:
            import torch
            from openpi_vega3d.towers import TOWER_REGISTRY
            from openpi_vega3d.towers.diagnostics import run_tower_on_obs_head

            log_phase(
                "tower",
                "GEN_TOWER smoke: type=%s ckpt=%s device=%s",
                args.gen_tower,
                args.gen_tower_ckpt,
                args.device,
            )
            torch_device = torch.device(args.device)
            gt_kwargs: dict = {"checkpoint_dir": args.gen_tower_ckpt}
            if args.gen_tower == "wan_t2v":
                gt_kwargs["prompt_emb_path"] = args.gen_tower_prompt_emb
            log_trace("11t0", "construct TOWER_REGISTRY[%s]", args.gen_tower)
            gen_tower = TOWER_REGISTRY[args.gen_tower](**gt_kwargs)
            gen_tower.to(torch_device)
            log_trace("11t1", "gen_tower on %s; check_output(head_rgb)", args.device)
            run_tower_on_obs_head(gen_tower, obs["head_rgb"], device=torch_device)
            log_trace("11t2", "gen_tower check_output complete")

        action_queue: deque[np.ndarray] = deque()
        total_reward = 0.0
        success = False
        step = 0
        t_start = time.perf_counter()
        replans = 0

        while step < max_steps:
            if len(action_queue) == 0:
                replans += 1
                log_phase("infer", "Replan #%d (queue empty) — policy.infer()", replans)
                t_inf0 = time.perf_counter()
                log_trace("12", "env step=%d replan=%d about to format_obs + policy.infer()", step, replans)
                batch = format_obs_for_policy(obs)
                result = policy.infer(batch)
                log_trace("13", "env step=%d replan=%d policy.infer() returned", step, replans)
                infer_s = time.perf_counter() - t_inf0
                actions_chunk = result["actions"]  # (action_horizon, 23)
                ac = np.asarray(actions_chunk)
                log_phase(
                    "infer",
                    "infer() done in %.2fs | actions shape=%s | min=%.4f max=%.4f mean=%.4f",
                    infer_s,
                    ac.shape,
                    float(np.min(ac)),
                    float(np.max(ac)),
                    float(np.mean(ac)),
                )
                n_take = min(args.chunk_size, len(actions_chunk))
                action_queue.extend(actions_chunk[:n_take])
                _LOG.debug("Queued %d actions for execution", n_take)

            action = action_queue.popleft()
            if step == 0:
                log_trace("14", "about to first env.step() after reset")
            obs, reward, terminated, truncated, info = env.step(action)
            if step == 0:
                log_trace("15", "first env.step() returned")

            total_reward += reward
            step += 1

            if step <= 10 or step % 100 == 0:
                log_phase(
                    "step",
                    "%5d | r=%+.4f total=%.4f | action |mean|=%.4f | info=%s",
                    step,
                    reward,
                    total_reward,
                    float(np.abs(action).mean()),
                    info,
                )

            if info.get("success", False):
                success = True
                log_phase("rollout", "SUCCESS at step %d", step)
                break

            if terminated or truncated:
                log_phase("rollout", "Episode ended terminated=%s truncated=%s at step %d", terminated, truncated, step)
                break

        elapsed = time.perf_counter() - t_start
        summary["status"] = "finished"
        summary["steps"] = step
        summary["replans"] = replans
        summary["success"] = success
        summary["total_reward"] = float(total_reward)
        summary["wall_time_s"] = float(elapsed)
        summary["wall_time_rollout_s"] = float(time.perf_counter() - t_roll0)
        summary["ms_per_step"] = float(elapsed / max(step, 1) * 1000)
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

        _LOG.info("=" * 60)
        _LOG.info("Episode finished:")
        _LOG.info("  Steps:        %d", step)
        _LOG.info("  Replans:      %d", replans)
        _LOG.info("  Total reward: %.4f", total_reward)
        _LOG.info("  Success:      %s", success)
        _LOG.info("  Wall time:    %.1fs (%.0f ms/step)", elapsed, elapsed / max(step, 1) * 1000)
        _LOG.info("=" * 60)

        log_trace("16", "about to env.close() (OmniGibson shutdown if app running)")
        env.close()
        log_trace("17", "env.close() returned OK")
    except BaseException as e:
        summary["status"] = "error"
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _LOG.exception("Rollout failed: %s", e)
        raise
    finally:
        if json_out and not args.no_json_out:
            try:
                p = Path(json_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=str)
                _LOG.info("Wrote summary JSON: %s", p.resolve())
            except OSError as e:
                _LOG.warning("Could not write JSON summary to %s: %s", json_out, e)


if __name__ == "__main__":
    main()
