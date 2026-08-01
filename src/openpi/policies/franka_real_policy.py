"""pi0.5 transforms for the real single-arm Franka (FR3) teleop demos.

Dataset contract, as produced by multi-fast's scripts/real/convert_real_to_lerobot.py:

    image          (224,224,3) uint8    scene view    <- cam_6_scene
    wrist_image    (224,224,3) uint8    wrist view A  <- cam_3_wrist
    wrist_image_2  (224,224,3) uint8    wrist view B  <- cam_4_wrist
    state          (8,)        float32  7 FR3 joint angles + gripper
    actions        (7,)        float32  [dx, dy, dz, rx, ry, rz, grip], rotation as rotvec

State is raw joint angles rather than EEF pose: it is what the demos actually record,
it matches pi0.5-base's pretraining convention (dims 0-5 are joint angles), and it
avoids an FK dependency that the openpi training container deliberately lacks.

The source demos store rotation as a quaternion plus kp/kd impedance channels. That
conversion lives in the converter so this module stays a thin key remap.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_real_example() -> dict:
    """Creates a random input example for the real-Franka policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaRealInputs(transforms.DataTransformFn):
    """Maps the converted real-Franka dataset into pi0.5's input dict."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/wrist_image_2"]),
            },
            # All three views are real cameras, so nothing is masked off here. LIBERO
            # zero-pads and masks the right wrist because it only has two views.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaRealOutputs(transforms.DataTransformFn):
    """Strips the model's 32-dim action padding back to the dataset's action dim."""

    # 7 for [dx, dy, dz, rx, ry, rz, grip]; 9 if the converter kept the kp/kd channels.
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
