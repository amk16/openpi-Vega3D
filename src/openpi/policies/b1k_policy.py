import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES


def make_b1k_example() -> dict:
    """Creates a random input example for the B1K policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data):
    """Extract the 23-dim state vector from the full 256-dim proprioception.

    Assumes perfect correlation for the two gripper fingers (summed to single width).
    """
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  # 7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  # 7
    left_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        arm_right_qpos,
        left_gripper_width,
        right_gripper_width,
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action = data["actions"]

        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (True, True, True)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (True, True, True)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "task_index" in data:
            inputs["task_index"] = data["task_index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
