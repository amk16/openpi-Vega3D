"""Minimal OmniGibson environment wrapper for BEHAVIOR inference rollouts.

Replaces the RL-specific BehaviorEnv from RLinf with a thin wrapper that
only handles observation extraction and action validation -- no PPO buffers,
value heads, reward shaping, crash recovery, or diagnostic logging.
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from omnigibson.envs import VectorEnvironment
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
    generate_basic_environment_config,
)
from omnigibson.learning.wrappers.rgb_low_res_wrapper import RGBLowResWrapper
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch

try:
    from gello.robots.sim_robot.og_teleop_utils import (
        augment_rooms,
        generate_robot_config,
        get_task_relevant_room_types,
        load_available_tasks,
    )
    from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES

    GELLO_AVAILABLE = True
except ImportError:
    GELLO_AVAILABLE = False
    DISABLED_TRANSITION_RULES = []

logger = logging.getLogger(__name__)
# Same logger name as scripts/run_rollout.py so trace lines land in one file.
_rollout_trace = logging.getLogger("run_rollout")


def _flush_runrollout_logs() -> None:
    """Best-effort flush so the last trace survives native crashes."""
    for h in logging.root.handlers:
        flush = getattr(h, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
    sys.stdout.flush()
    sys.stderr.flush()


def _trace_env(msg: str, *args) -> None:
    # WARNING: OmniGibson sets root log level to WARNING after Kit launch; INFO would be dropped.
    _rollout_trace.warning("[trace env] " + msg, *args)
    _flush_runrollout_logs()

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

ROBOT_NAME = "robot_r1"


class SimpleEnv:
    """Thin wrapper around OmniGibson's VectorEnvironment for inference-only rollouts.

    Observation contract (matches BehaviorEnv):
        head_rgb:         np.ndarray (H, W, 3) uint8
        left_wrist_rgb:   np.ndarray (H, W, 3) uint8
        right_wrist_rgb:  np.ndarray (H, W, 3) uint8
        proprio:          np.ndarray (N,) float
        task_description: str
        task_index:       int
    """

    def __init__(
        self,
        task_name: str,
        task_description: str,
        *,
        omnigibson_cfg: dict | None = None,
        partial_scene_load: bool = False,
        disable_transition_rules: bool = False,
    ):
        _trace_env(
            "E00 SimpleEnv.__init__ start task=%s partial_scene=%s disable_rules=%s",
            task_name,
            partial_scene_load,
            disable_transition_rules,
        )
        self.task_name = task_name
        self.task_description = task_description
        self.task_index = {v: k for k, v in TASK_INDICES_TO_NAMES.items()}[task_name]

        if disable_transition_rules and DISABLED_TRANSITION_RULES:
            for rule in DISABLED_TRANSITION_RULES:
                rule.ENABLED = False

        if omnigibson_cfg is not None:
            self._og_cfg = OmegaConf.to_container(
                OmegaConf.create(omnigibson_cfg), resolve=True
            ) if not isinstance(omnigibson_cfg, dict) else omnigibson_cfg
        else:
            self._og_cfg = self._build_og_cfg(partial_scene_load)

        self._og_cfg["task"]["activity_name"] = task_name

        _trace_env("E01 about to VectorEnvironment(1, cfg) — Isaac Sim / Kit startup")
        self.env = VectorEnvironment(1, self._og_cfg)
        _trace_env("E02 VectorEnvironment returned; %d sub-env(s)", len(self.env.envs))
        for i, sub_env in enumerate(self.env.envs):
            _trace_env(
                "E03 RGBLowResWrapper sub_env %d/%d",
                i + 1,
                len(self.env.envs),
            )
            RGBLowResWrapper(sub_env)
        _trace_env("E04 all RGBLowResWrapper(s) applied")

        self._max_steps = self._og_cfg.get("task", {}).get(
            "termination_config", {}
        ).get("max_steps", 5000)

        self._step_count = 0
        self._last_actions = None
        _trace_env("E05 SimpleEnv.__init__ complete")

    def _build_og_cfg(self, partial_scene_load: bool) -> dict:
        if not GELLO_AVAILABLE:
            raise RuntimeError(
                "No omnigibson_cfg provided and gello helpers unavailable. "
                "Either pass omnigibson_cfg or install gello."
            )

        available_tasks = load_available_tasks()
        if self.task_name not in available_tasks:
            raise ValueError(f"Task {self.task_name} not in available_tasks")

        task_cfg = available_tasks[self.task_name][0]
        og_cfg = generate_basic_environment_config(
            task_name=self.task_name, task_cfg=task_cfg
        )

        if partial_scene_load:
            rooms = get_task_relevant_room_types(activity_name=self.task_name)
            rooms = augment_rooms(rooms, task_cfg["scene_model"], self.task_name)
            og_cfg["scene"]["load_room_types"] = rooms

        og_cfg["robots"] = [
            generate_robot_config(task_name=self.task_name, task_cfg=task_cfg)
        ]
        og_cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        og_cfg["robots"][0]["proprio_obs"] = list(
            PROPRIOCEPTION_INDICES["R1Pro"].keys()
        )
        og_cfg["robots"][0]["name"] = ROBOT_NAME
        og_cfg["task"]["include_obs"] = False

        return og_cfg

    def _extract_obs(self, raw_obs_list: list[dict]) -> dict:
        """Convert raw OmniGibson observation into a clean dict."""
        raw_obs = raw_obs_list[0]

        left_rgb = right_rgb = head_rgb = proprio = None
        for sensor_data in raw_obs.values():
            if not isinstance(sensor_data, dict):
                continue
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_rgb = v["rgb"]
                elif "right_realsense_link:Camera:0" in k:
                    right_rgb = v["rgb"]
                elif "zed_link:Camera:0" in k:
                    head_rgb = v["rgb"]
                elif "proprio" in k:
                    proprio = v

        missing = [
            n
            for n, x in [
                ("left_rgb", left_rgb),
                ("right_rgb", right_rgb),
                ("head_rgb", head_rgb),
                ("proprio", proprio),
            ]
            if x is None
        ]
        if missing:
            raise KeyError(f"Missing observation keys: {missing}")

        def _to_uint8_hwc(img):
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return img[..., :3]

        return {
            "head_rgb": _to_uint8_hwc(head_rgb),
            "left_wrist_rgb": _to_uint8_hwc(left_rgb),
            "right_wrist_rgb": _to_uint8_hwc(right_rgb),
            "proprio": np.asarray(proprio, dtype=np.float64),
            "task_description": self.task_description,
            "task_index": self.task_index,
        }

    @staticmethod
    def _validate_and_clip_actions(actions: np.ndarray) -> np.ndarray:
        """Clamp to [-1, 1] and replace NaN/inf."""
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(actions, -1.0, 1.0)

    def load_task_instance(self, instance_id: int) -> None:
        """Load a specific task instance (TRO state) into the environment."""
        import omnigibson as og

        curr_env = self.env.envs[0]
        task = curr_env.task
        scene_model = task.scene_name

        robot = curr_env.scene.object_registry("name", ROBOT_NAME)
        if robot is None:
            raise RuntimeError("Robot not found in scene")

        with og.sim.stopped():
            robot.base_footprint_link.mass = 250.0

        tro_filename = task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=task.activity_name,
            activity_definition_id=task.activity_definition_id,
            activity_instance_id=instance_id,
        )

        tro_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.task_name}_instances/{tro_filename}-tro_state.json",
        )

        if not os.path.exists(tro_path):
            raise FileNotFoundError(f"TRO state file not found: {tro_path}")

        with open(tro_path) as f:
            tro_state = recursively_convert_to_torch(json.load(f))

        for tro_key, state_data in tro_state.items():
            if tro_key == "robot_poses":
                robot_pos = state_data[robot.model_name][0]["position"]
                robot_quat = state_data[robot.model_name][0]["orientation"]
                robot.set_position_orientation(robot_pos, robot_quat)
                curr_env.scene.write_task_metadata(key=tro_key, data=state_data)
            else:
                task.object_scope[tro_key].load_state(state_data, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        curr_env.scene.update_initial_file()
        curr_env.scene.reset()

        logger.info(
            f"Loaded instance {instance_id} for task {self.task_name} (idx={self.task_index})"
        )

    def reset(self) -> dict:
        """Reset environment and return clean observation dict."""
        raw_obs, _ = self.env.reset()
        self._step_count = 0
        self._last_actions = None
        return self._extract_obs(raw_obs)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one action and return (obs, reward, terminated, truncated, info).

        Args:
            action: (23,) array of joint commands in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = self._validate_and_clip_actions(action)

        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        raw_obs, _rewards, terminations, truncations, infos = self.env.step(
            action_tensor
        )

        self._step_count += 1
        self._last_actions = action.copy()

        obs = self._extract_obs(raw_obs)

        terminated = bool(terminations[0]) if hasattr(terminations, "__getitem__") else bool(terminations)
        truncated = bool(truncations[0]) if hasattr(truncations, "__getitem__") else bool(truncations)

        info_dict = infos[0] if isinstance(infos, list) else infos
        success = False
        if isinstance(info_dict, dict):
            success = info_dict.get("done", {}).get("success", False)

        return obs, 0.0, terminated, truncated, {"success": success, "step": self._step_count}

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def close(self):
        """Tear down Kit; VectorEnvironment/Environment.close() are no-ops upstream."""
        _trace_env("E99 SimpleEnv.close() start")
        if hasattr(self, "env"):
            self.env.close()
        import omnigibson as og

        if og.app is not None:
            _trace_env(
                "E99b calling omnigibson.shutdown() (avoids exit-time segfault from stale Kit)"
            )
            og.shutdown()
        _trace_env("E99z SimpleEnv.close() done")
