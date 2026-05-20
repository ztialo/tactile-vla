#!/usr/bin/env python3
"""Attach-only FR3 teleoperation controller driven by OmniGraph ROS2 Bridge data.

This script is intended to run *inside an already running Isaac Sim session*.
It does not import or use rclpy.
"""

from __future__ import annotations

LOG = True

import csv
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any, Optional, Sequence, Tuple
import numpy as np

# Lazy-loaded Isaac/Omni modules. This keeps import-time failurs informative when
# the script is executed outside a running Isaac Sim process.
og = None
omni = None
carb = None
omni_appwindow = None
Usd = None
UsdGeom = None
World = None
SingleArticulation = None
XFormPrim = None
ArticulationAction = None
create_prim = None
is_prim_path_valid = None

# plot configure
FT_AXIS_TO_INDEX = {
    "fx": 0,
    "fy": 1,
    "fz": 2,
    "tx": 3,
    "ty": 4,
    "tz": 5,
}
# LIVE_FT_PLOT_AXES = ("fx", "fy", "fz", "tx", "ty", "tz")  # Set to one or more of: fx, fy, fz, tx, ty, tz
LIVE_FT_PLOT_AXES = ("fx", "fy", "fz")
LIVE_FT_PLOT_AXIS = None  # Legacy single-axis fallback
LIVE_FT_PLOT_UPDATE_HZ = 8.0
LIVE_FT_PLOT_WINDOW_SECONDS = 20.0
LIVE_FT_FORCE_Y_LIMIT = 20.0
LIVE_FT_TORQUE_Y_LIMIT = 0.5


# Robot and target configuration.
ROBOT_PRIM_PATH = "/Root/fr3_ft_wc/fr3"
TARGET_PRIM_PATH = "/World/rmp_target"
TARGET_INITIAL_POSITION = [0.45, 0.0, 0.45]
ENABLE_TARGET_VISUALIZATION = False
KEYBOARD_TARGET_PRIM_PATH = "/Root/target"
REPO_ROOT = Path("/home/zdli/Projects/tactile-vla")
# REPO_ROOT =  Path("/home/tactilemanipulationlab/lerobot/teleop_ws/src/teleop_starai_franka")
LOG_DIR = REPO_ROOT / "ft-log"

# OmniGraph ROS subscribe node configuration.
# Edit these paths if your Action Graph/node names or output attribute names differ.
ACTION_GRAPH_PATH = "/Root/eef_pose_sub"
ROS2_SUBSCRIBE_POSE_NODE_NAME = "ros2_subscribe_eef_pose"
GRIPPER_ACTION_GRAPH_PATH = "/Root/gripper_controller"
ROS2_SUBSCRIBE_GRIPPER_NODE_NAME = "ros2_subscribe_gripper_command"
DIFF_IK_METHOD = "dls"  # "dls", "pinv", or "transpose"
DIFF_IK_DAMPING = 0.05
DIFF_IK_POSITION_GAIN = 1.0
DIFF_IK_ORIENTATION_GAIN = 1.0

# Relative teleop mapping:
# commanded_pose = home_pose (+) delta(input_pose relative to first received pose)
USE_RELATIVE_INPUT = True
EE_HOME_REFERENCE_PRIM_PATH = "/Root/fr3_ft_wc/fr3/fr3_hand_tcp"
EE_HOME_REFERENCE_PRIM_FALLBACK_PATHS = []
DIFF_IK_EE_PRIM_CANDIDATES = [
    EE_HOME_REFERENCE_PRIM_PATH,
    "/Root/fr3_ft_wc/fr3/fr3_link8",
    "/Root/fr3_ft_wc/fr3/fr3_hand",
    "/Root/fr3_ft_wc/fr3/fr3_link7",
]
DIFF_IK_EE_BODY_CANDIDATES = [
    # "fr3_hand_tcp",
    "fr3_link8",
    # "fr3_hand",
    # "fr3_link7",
]
ARM_JOINT_NAME_CANDIDATES = tuple(f"fr3_joint{i}" for i in range(1, 8))
POSITION_DELTA_SCALE = 3.0
KEYBOARD_POSITION_STEP_M = 0.01
KEYBOARD_ROTATION_STEP_RAD = math.radians(5.0)
KEYBOARD_FORWARD_KEYS = ("PERIOD", "DOT", "GREATER", "GREATERTHAN", ">")
KEYBOARD_BACKWARD_KEYS = ("COMMA", "LESS", "LESSTHAN", "<")
TRANSLATION_MAPPING_MODE = "world_delta"  # "world_delta" (default) or "local_delta"
# Orientation mapping options:
# 1) delta_q = q_current * conj(q_ref), cmd_q = delta_q * q_home
# 2) delta_q = conj(q_ref) * q_current, cmd_q = q_home * delta_q
ORIENTATION_MAPPING_MODE = "current_conjref__delta_home"  # or "conjref_current__home_delta"
SEND_ORIENTATION_TARGET = True
# Incoming quaternion order for 4-element sequence outputs from OmniGraph.
# ROS geometry_msgs uses (x, y, z, w). Isaac Core APIs expect (w, x, y, z).
OG_SEQUENCE_QUAT_ORDER = "xyzw"  # or "wxyz" if your node already outputs scalar-first
# For ROS2 Subscribe Pose/PoseStamped nodes, output names can vary by node type/version.
# Keep these candidates editable and ordered by preference.
OG_POSITION_OUTPUT_ATTR_CANDIDATES = [
    "outputs:position",
    "outputs:pose:position",
]
OG_ORIENTATION_OUTPUT_ATTR_CANDIDATES = [
    "outputs:orientation",
    "outputs:pose:orientation",
]
OG_POSE_OUTPUT_ATTR_CANDIDATES = [
    "outputs:pose",
    "outputs:data",
    "outputs:message",
    "outputs:data:pose",
]
OG_GRIPPER_OUTPUT_ATTR_CANDIDATES = [
    "outputs:position",
    "outputs:message:position",
    "outputs:data:position",
    "outputs:message",
    "outputs:data",
]
FR3_HOME_JOINT_TARGETS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.356,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.571,
    "fr3_joint7": 0.785,
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}
FR3_HOME_SETTLE_SECONDS = 0.5
GRIPPER_JOINT_NAMES = ("fr3_finger_joint1", "fr3_finger_joint2")
GRIPPER_MIN_POS = 0.0
GRIPPER_MAX_POS = 0.04
GRIPPER_MAX_SPEED = 0.03  # finger joint position units per second
KEYBOARD_GRIPPER_COARSE_STEP = 0.001
KEYBOARD_GRIPPER_STEP = 0.0002
LEFT_FT_JOINT_NAME_CANDIDATES = ("fr3_left_ft_pad", "fr3_left_ft", "fr3_gripper_left_ft")
RIGHT_FT_JOINT_NAME_CANDIDATES = ("fr3_right_ft_pad", "fr3_right_ft", "fr3_gripper_right_ft")
LEFT_FT_JOINT_PRIM_PATH = "/Root/fr3_ft_wc/fr3/fr3_left_ft_pad"
RIGHT_FT_JOINT_PRIM_PATH = "/Root/fr3_ft_wc/fr3/fr3_right_ft_pad"
GRIPPER_CONTACT_FORCE_THRESHOLD = 2.0
FT_FORCE_ATTR_CANDIDATES = [
    "body_incoming_joint_wrench_b",
    "state:normalForce",
    "state:force",
    "state:linearForce",
    "physxJoint:normalForce",
    "physxJoint:force",
]


def _ensure_isaac_imports() -> None:
    global og, omni, carb, omni_appwindow, Usd, UsdGeom, World, SingleArticulation, XFormPrim, ArticulationAction, create_prim, is_prim_path_valid
    if og is not None:
        return
    try:
        import carb as _carb
        import omni as _omni
        import omni.appwindow as _omni_appwindow
        import omni.graph.core as _og
        from pxr import Usd as _Usd
        from pxr import UsdGeom as _UsdGeom
        from isaacsim.core.api import World as _World
        from isaacsim.core.prims import SingleArticulation as _SingleArticulation
        from isaacsim.core.prims import XFormPrim as _XFormPrim
        from isaacsim.core.utils.types import ArticulationAction as _ArticulationAction
        from isaacsim.core.utils.prims import create_prim as _create_prim
        from isaacsim.core.utils.prims import is_prim_path_valid as _is_prim_path_valid
    except Exception as exc:
        raise RuntimeError(
            "Isaac Sim Python modules are unavailable in this process. "
            "Run this script inside an already running Isaac Sim app (Script Editor/extension), "
            "not as a standalone `./python.sh script.py` process."
        ) from exc

    carb = _carb
    omni = _omni
    omni_appwindow = _omni_appwindow
    og = _og
    Usd = _Usd
    UsdGeom = _UsdGeom
    World = _World
    SingleArticulation = _SingleArticulation
    XFormPrim = _XFormPrim
    ArticulationAction = _ArticulationAction
    create_prim = _create_prim
    is_prim_path_valid = _is_prim_path_valid


class FR3DifferentialIKController:
    def __init__(self, name: str, robot_articulation: Any, physics_dt: float = 1.0 / 60.0) -> None:
        del name, physics_dt
        self._robot_articulation = robot_articulation
        self._arm_joint_indices: Optional[np.ndarray] = None
        self._ee_body_index: Optional[int] = None
        self._joint_lower_limits: Optional[np.ndarray] = None
        self._joint_upper_limits: Optional[np.ndarray] = None
        self._ee_prim_path: Optional[str] = None

    def reset(self) -> None:
        self._arm_joint_indices = None
        self._ee_body_index = None
        self._joint_lower_limits = None
        self._joint_upper_limits = None
        self._ee_prim_path = None

    def _ensure_initialized(self) -> None:
        if self._arm_joint_indices is not None and self._ee_body_index is not None and self._ee_prim_path is not None:
            return

        dof_names = list(getattr(self._robot_articulation, "dof_names", []) or [])
        if not dof_names:
            raise RuntimeError("Differential IK controller could not resolve articulation DOF names.")
        self._arm_joint_indices = np.asarray(
            [dof_names.index(joint_name) for joint_name in ARM_JOINT_NAME_CANDIDATES],
            dtype=np.int64,
        )

        dof_props = getattr(self._robot_articulation, "dof_properties", None)
        if dof_props is not None:
            self._joint_lower_limits = np.asarray(dof_props["lower"], dtype=np.float64)[self._arm_joint_indices]
            self._joint_upper_limits = np.asarray(dof_props["upper"], dtype=np.float64)[self._arm_joint_indices]

        articulation_view = getattr(self._robot_articulation, "_articulation_view", None)
        if articulation_view is None:
            raise RuntimeError("Differential IK controller could not access articulation view for Jacobians.")

        available_body_names = set(getattr(articulation_view, "body_names", []) or [])
        for body_name in DIFF_IK_EE_BODY_CANDIDATES:
            if body_name in available_body_names:
                self._ee_body_index = int(articulation_view.get_body_index(body_name))
                break
        if self._ee_body_index is None:
            raise RuntimeError(
                f"Differential IK controller could not resolve an end-effector body. Available bodies: {sorted(available_body_names)}"
            )

        for prim_path in DIFF_IK_EE_PRIM_CANDIDATES:
            if is_prim_path_valid(prim_path):
                self._ee_prim_path = prim_path
                break
        if self._ee_prim_path is None:
            raise RuntimeError(
                f"Differential IK controller could not resolve an end-effector prim from {DIFF_IK_EE_PRIM_CANDIDATES}."
            )

    def forward(self, target_end_effector_position: np.ndarray, target_end_effector_orientation: np.ndarray):
        self._ensure_initialized()

        current_joint_positions = np.asarray(self._robot_articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
        arm_joint_positions = current_joint_positions[self._arm_joint_indices]

        jacobians = np.asarray(self._robot_articulation._articulation_view.get_jacobians(), dtype=np.float64)
        ee_jacobian = jacobians[0, self._ee_body_index - 1, :, self._arm_joint_indices]
        ee_jacobian = _normalize_diff_ik_jacobian(
            ee_jacobian,
            expected_task_dim=6,
            expected_joint_dim=int(self._arm_joint_indices.size),
        )

        current_ee_position, current_ee_orientation = _get_prim_world_pose(self._ee_prim_path)
        if current_ee_position is None or current_ee_orientation is None:
            raise RuntimeError(f"Failed to read end-effector pose from '{self._ee_prim_path}'.")

        task_error = _compute_diff_ik_error(
            current_position=current_ee_position,
            current_orientation=current_ee_orientation,
            goal_position=np.asarray(target_end_effector_position, dtype=np.float64),
            goal_orientation=(
                None
                if target_end_effector_orientation is None
                else _quat_normalize_wxyz(np.asarray(target_end_effector_orientation, dtype=np.float64))
            ),
        )
        if target_end_effector_orientation is None:
            ee_jacobian = ee_jacobian[:3, :]

        delta_joint_positions = _solve_diff_ik_delta(ee_jacobian, task_error)
        joint_position_targets = arm_joint_positions + delta_joint_positions

        if self._joint_lower_limits is not None and self._joint_upper_limits is not None:
            joint_position_targets = np.clip(joint_position_targets, self._joint_lower_limits, self._joint_upper_limits)

        return ArticulationAction(
            joint_positions=np.asarray(joint_position_targets, dtype=np.float32),
            joint_indices=np.asarray(self._arm_joint_indices, dtype=np.int32),
        )


def _normalize_diff_ik_jacobian(
    jacobian: np.ndarray,
    expected_task_dim: int,
    expected_joint_dim: int,
) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2:
        raise RuntimeError(
            f"Differential IK expected a 2D Jacobian, got shape {jacobian.shape}."
        )

    if jacobian.shape == (expected_task_dim, expected_joint_dim):
        return jacobian
    if jacobian.shape == (expected_joint_dim, expected_task_dim):
        return jacobian.T

    raise RuntimeError(
        "Differential IK received an unexpected Jacobian shape "
        f"{jacobian.shape}; expected ({expected_task_dim}, {expected_joint_dim}) "
        f"or ({expected_joint_dim}, {expected_task_dim})."
    )


def _as_xyz(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) == 3:
        xyz = [float(value[0]), float(value[1]), float(value[2])]
    elif all(hasattr(value, attr) for attr in ("x", "y", "z")):
        xyz = [float(value.x), float(value.y), float(value.z)]
    else:
        return None
    if not all(math.isfinite(v) for v in xyz):
        return None
    return xyz


def _as_xyzw(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, Sequence) and len(value) == 4:
        xyzw = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    elif all(hasattr(value, attr) for attr in ("x", "y", "z", "w")):
        xyzw = [float(value.x), float(value.y), float(value.z), float(value.w)]
    elif hasattr(value, "GetImaginary") and hasattr(value, "GetReal"):
        imag = value.GetImaginary()
        xyzw = [float(imag[0]), float(imag[1]), float(imag[2]), float(value.GetReal())]
    else:
        return None
    if not all(math.isfinite(v) for v in xyzw):
        return None
    return xyzw


def _xyzw_to_wxyz(xyzw: Sequence[float]) -> list[float]:
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _quat_conj_wxyz(q: Sequence[float]) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul_wxyz(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_normalize_wxyz(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm <= 0.0 or not np.isfinite(norm):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q_arr / norm


def _quat_to_rotmat_wxyz(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in _quat_normalize_wxyz(q)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle_wxyz(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis_arr))
    if axis_norm <= 0.0 or not np.isfinite(axis_norm):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half_angle = 0.5 * float(angle_rad)
    scale = math.sin(half_angle) / axis_norm
    return _quat_normalize_wxyz(
        np.asarray(
            [
                math.cos(half_angle),
                axis_arr[0] * scale,
                axis_arr[1] * scale,
                axis_arr[2] * scale,
            ],
            dtype=np.float64,
        )
    )


def _fmt_optional_scalar(value: Optional[float], decimals: int = 6) -> str:
    if value is None:
        return ""
    scalar = float(value)
    return f"{scalar:.{decimals}f}" if math.isfinite(scalar) else ""


def _fmt_optional_scalar(value: Optional[float], decimals: int = 6) -> str:
    if value is None:
        return ""
    scalar = float(value)
    return f"{scalar:.{decimals}f}" if math.isfinite(scalar) else ""


def _quat_error_as_rotvec_wxyz(goal: Sequence[float], current: Sequence[float]) -> np.ndarray:
    q_err = _quat_normalize_wxyz(_quat_mul_wxyz(goal, _quat_conj_wxyz(current)))
    if float(q_err[0]) < 0.0:
        q_err = -q_err
    vec = np.asarray(q_err[1:], dtype=np.float64)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vec_norm, float(q_err[0]))
    return (angle / vec_norm) * vec


def _compute_diff_ik_error(
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    goal_position: np.ndarray,
    goal_orientation: Optional[np.ndarray],
) -> np.ndarray:
    position_error = float(DIFF_IK_POSITION_GAIN) * (
        np.asarray(goal_position, dtype=np.float64) - np.asarray(current_position, dtype=np.float64)
    )
    if goal_orientation is None:
        return position_error
    orientation_error = float(DIFF_IK_ORIENTATION_GAIN) * _quat_error_as_rotvec_wxyz(
        np.asarray(goal_orientation, dtype=np.float64),
        np.asarray(current_orientation, dtype=np.float64),
    )
    return np.concatenate([position_error, orientation_error], axis=0)


def _solve_diff_ik_delta(jacobian: np.ndarray, task_error: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    task_error = np.asarray(task_error, dtype=np.float64).reshape(-1, 1)
    if DIFF_IK_METHOD == "pinv":
        return (np.linalg.pinv(jacobian) @ task_error).reshape(-1)
    if DIFF_IK_METHOD == "transpose":
        return (jacobian.T @ task_error).reshape(-1)
    transpose = jacobian.T
    damping = float(DIFF_IK_DAMPING)
    lambda_matrix = np.eye(jacobian.shape[0], dtype=np.float64) * (damping**2)
    return (transpose @ np.linalg.inv(jacobian @ transpose + lambda_matrix) @ task_error).reshape(-1)


def _get_prim_world_pose(prim_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None, None
        world_mtx = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_mtx.ExtractTranslation()
        rotation = world_mtx.ExtractRotationQuat()
        position = np.asarray([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float64)
        orientation = _quat_normalize_wxyz(
            np.asarray(
                [
                    float(rotation.GetReal()),
                    float(rotation.GetImaginary()[0]),
                    float(rotation.GetImaginary()[1]),
                    float(rotation.GetImaginary()[2]),
                ],
                dtype=np.float64,
            )
        )
        return position, orientation
    except Exception:
        return None, None


def _as_wxyz(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if hasattr(value, "GetImaginary") and hasattr(value, "GetReal"):
        # USD GfQuat stores scalar separately; convert directly to Isaac scalar-first order.
        imag = value.GetImaginary()
        wxyz = [float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
    elif all(hasattr(value, attr) for attr in ("x", "y", "z", "w")):
        # Pose/PoseStamped style orientation fields.
        wxyz = [float(value.w), float(value.x), float(value.y), float(value.z)]
    else:
        xyzw = _as_xyzw(value)
        if xyzw is None:
            return None
        if OG_SEQUENCE_QUAT_ORDER == "wxyz":
            wxyz = [float(xyzw[0]), float(xyzw[1]), float(xyzw[2]), float(xyzw[3])]
        else:
            wxyz = _xyzw_to_wxyz(xyzw)
    if not all(math.isfinite(v) for v in wxyz):
        return None
    return wxyz


def _read_og_attr(attr_path: str) -> Any:
    try:
        return og.Controller.get(attr_path)
    except Exception:
        pass
    try:
        attr = og.Controller.attribute(attr_path)
        return og.Controller.get(attr)
    except Exception:
        return None


def _read_og_leaf_scalar(base_attr_path: str, component: str) -> Optional[float]:
    for attr_path in (f"{base_attr_path}:{component}", f"{base_attr_path}.{component}"):
        value = _read_og_attr(attr_path)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _fmt_seq(value: Sequence[float] | np.ndarray, decimals: int = 3) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr) + "]"


def _extract_pose_from_struct(raw_pose: Any) -> Optional[Tuple[list[float], list[float]]]:
    if raw_pose is None:
        return None

    # PoseStamped-like container.
    if hasattr(raw_pose, "pose"):
        pose_obj = raw_pose.pose
        pos = _as_xyz(getattr(pose_obj, "position", None))
        ori = _as_wxyz(getattr(pose_obj, "orientation", None))
        if pos is not None and ori is not None:
            return pos, ori

    # Pose-like container.
    pos = _as_xyz(getattr(raw_pose, "position", None))
    ori = _as_wxyz(getattr(raw_pose, "orientation", None))
    if pos is not None and ori is not None:
        return pos, ori
    return None


def _extract_gripper_width(raw_msg: Any) -> Optional[float]:
    if raw_msg is None:
        return None

    names = None
    positions = None

    if hasattr(raw_msg, "name"):
        names = getattr(raw_msg, "name", None)
    if hasattr(raw_msg, "position"):
        positions = getattr(raw_msg, "position", None)

    if isinstance(raw_msg, dict):
        names = raw_msg.get("name", names)
        positions = raw_msg.get("position", positions)

    if positions is None and isinstance(raw_msg, Sequence) and not isinstance(raw_msg, (str, bytes)):
        if len(raw_msg) > 0:
            try:
                val = float(raw_msg[0])
                return val if math.isfinite(val) else None
            except Exception:
                return None

    if positions is not None and isinstance(positions, Sequence) and len(positions) > 0:
        if names is not None and isinstance(names, Sequence):
            for joint_name in ("gripper_joint", "fr3_finger_joint1", "fr3_finger_joint2"):
                try:
                    idx = list(names).index(joint_name)
                    val = float(positions[idx])
                    return val if math.isfinite(val) else None
                except Exception:
                    continue
        try:
            val = float(positions[0])
            return val if math.isfinite(val) else None
        except Exception:
            return None

    try:
        val = float(raw_msg)
        return val if math.isfinite(val) else None
    except Exception:
        return None


class FrankaTeleopAttachRuntime:
    def __init__(self) -> None:
        node_path = f"{ACTION_GRAPH_PATH}/{ROS2_SUBSCRIBE_POSE_NODE_NAME}"
        self._node_path = node_path
        self._position_attr_paths = [
            f"{node_path}.{attr_name}" for attr_name in OG_POSITION_OUTPUT_ATTR_CANDIDATES
        ]
        self._orientation_attr_paths = [
            f"{node_path}.{attr_name}" for attr_name in OG_ORIENTATION_OUTPUT_ATTR_CANDIDATES
        ]
        self._pose_attr_paths = [f"{node_path}.{attr_name}" for attr_name in OG_POSE_OUTPUT_ATTR_CANDIDATES]
        gripper_node_path = f"{GRIPPER_ACTION_GRAPH_PATH}/{ROS2_SUBSCRIBE_GRIPPER_NODE_NAME}"
        self._gripper_attr_paths = [f"{gripper_node_path}.{attr_name}" for attr_name in OG_GRIPPER_OUTPUT_ATTR_CANDIDATES]

        self._fr3: Optional[Any] = None
        self._target: Optional[Any] = None
        self._controller: Optional[Any] = None
        self._articulation_controller = None
        self._world: Optional[Any] = None
        self._robot_ready = False

        self._physics_sub = None
        self._update_sub = None
        self._timeline_sub = None

        self._reset_needed = True
        self._warned_waiting_for_pose = False
        self._missing_pose_steps = 0
        self._warned_articulation_init = False
        self._latest_target_pose: Optional[Tuple[list[float], list[float]]] = None
        self._home_cmd_position: Optional[np.ndarray] = None
        self._home_cmd_orientation: Optional[np.ndarray] = None
        self._input_ref_position: Optional[np.ndarray] = None
        self._input_ref_orientation: Optional[np.ndarray] = None
        self._warned_waiting_home_pose = False
        self._disable_target_updates = not ENABLE_TARGET_VISUALIZATION
        self._printed_initial_pose_snapshot = False
        self._homing_done = False
        self._homing_elapsed_s = 0.0
        self._home_joint_positions_cache: Optional[np.ndarray] = None
        self._home_joint_targets_available = True
        self._stopped = False
        self._latest_gripper_width: Optional[float] = None
        self._gripper_target_width: Optional[float] = float(GRIPPER_MAX_POS)
        self._gripper_applied_width: Optional[float] = None
        self._gripper_joint_indices: Optional[Tuple[int, int]] = None
        self._warned_gripper_joint_missing = False
        self._ft_selected_attr: dict[str, str] = {}
        self._warned_ft_unavailable = False
        self._gripper_contact_latched = False
        self._keyboard = None
        self._keyboard_sub = None
        self._input_iface = None
        self._keyboard_target_position: Optional[np.ndarray] = None
        self._keyboard_target_orientation: Optional[np.ndarray] = None
        self._ft_log_file = None
        self._ft_log_writer = None
        self._ft_log_path: Optional[Path] = None
        self._ft_link_ids: Optional[Tuple[int, int]] = None
        self._ft_link_names: Optional[Tuple[str, str]] = None
        self._warned_ft_joint_force_missing = False
        self._printed_ft_joint_resolution = False
        self._live_plot_enabled = False
        self._live_plot_axes: tuple[str, ...] = tuple()
        self._live_plot_axis_indices: dict[str, int] = {}
        self._live_plot_start_s: Optional[float] = None
        self._live_plot_last_draw_s = 0.0
        self._live_plot_times: deque[float] = deque()
        self._live_plot_left_values: dict[str, deque[float]] = {}
        self._live_plot_right_values: dict[str, deque[float]] = {}
        self._live_plot_plt = None
        self._live_plot_fig = None
        self._live_plot_axes_by_name: dict[str, Any] = {}
        self._live_plot_lines: dict[str, tuple[Any, Any]] = {}

    def _reset_cycle_state(self) -> None:
        self._reset_needed = True
        self._warned_waiting_for_pose = False
        self._missing_pose_steps = 0
        self._warned_articulation_init = False
        self._latest_target_pose = None
        self._warned_waiting_home_pose = False
        self._latest_gripper_width = None
        self._gripper_target_width = float(GRIPPER_MAX_POS)
        self._gripper_applied_width = None

    def _reset_relative_reference_state(self) -> None:
        self._home_cmd_position = None
        self._home_cmd_orientation = None
        self._input_ref_position = None
        self._input_ref_orientation = None
        self._printed_initial_pose_snapshot = False
        self._homing_done = False
        self._homing_elapsed_s = 0.0
        self._home_joint_positions_cache = None
        self._home_joint_targets_available = True
        self._gripper_joint_indices = None
        self._warned_gripper_joint_missing = False
        self._ft_selected_attr = {}
        self._warned_ft_unavailable = False
        self._gripper_contact_latched = False
        self._keyboard_target_position = None
        self._keyboard_target_orientation = None
        self._ft_link_ids = None
        self._ft_link_names = None
        self._warned_ft_joint_force_missing = False
        self._printed_ft_joint_resolution = False

    def start(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is currently open in Isaac Sim.")

        if not is_prim_path_valid(ROBOT_PRIM_PATH):
            raise RuntimeError(
                f"Robot prim not found at '{ROBOT_PRIM_PATH}'. "
                "Update ROBOT_PRIM_PATH in this script."
            )

        if ENABLE_TARGET_VISUALIZATION and not is_prim_path_valid(TARGET_PRIM_PATH):
            create_prim(TARGET_PRIM_PATH, "Xform", translation=TARGET_INITIAL_POSITION)
            print(
                f"[INFO] Created target prim at '{TARGET_PRIM_PATH}' with position {TARGET_INITIAL_POSITION}",
                flush=True,
            )

        self._world = World.instance()
        if self._world is None:
            self._world = World(stage_units_in_meters=1.0)

        run_tag = uuid.uuid4().hex[:8]
        self._fr3 = self._world.scene.add(SingleArticulation(prim_path=ROBOT_PRIM_PATH, name=f"fr3_ctrl_{run_tag}"))
        self._controller = FR3DifferentialIKController(
            name=f"fr3_diff_ik_{run_tag}",
            robot_articulation=self._fr3,
        )

        if ENABLE_TARGET_VISUALIZATION:
            try:
                self._target = self._world.scene.add(XFormPrim(TARGET_PRIM_PATH, name=f"rmp_target_{run_tag}"))
            except TypeError:
                self._target = self._world.scene.add(XFormPrim(TARGET_PRIM_PATH))
        else:
            self._target = None
        timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )
        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self._update_sub = update_stream.create_subscription_to_pop(self._on_update)
        self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )
        self._setup_keyboard_debug()
        self._open_ft_log()
        self._setup_live_ft_plot()

        print("[INFO] Attach runtime started.", flush=True)
        if not ENABLE_TARGET_VISUALIZATION:
            print("[INFO] Target visualization is disabled. Driving FR3 directly from teleop commands.", flush=True)

    def shutdown(self) -> None:
        self._stopped = True
        self._physics_sub = None
        self._update_sub = None
        self._timeline_sub = None
        self._teardown_keyboard_debug()
        self._close_ft_log()
        self._teardown_live_ft_plot()
        self._fr3 = None
        self._target = None
        self._controller = None
        self._articulation_controller = None
        self._world = None
        self._robot_ready = False
        self._disable_target_updates = not ENABLE_TARGET_VISUALIZATION
        self._reset_cycle_state()
        self._reset_relative_reference_state()

    def _open_ft_log(self) -> None:
        if not LOG:
            self._ft_log_file = None
            self._ft_log_writer = None
            self._ft_log_path = None
            return
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._ft_log_path = LOG_DIR / f"ft_log_{timestamp}.csv"
            self._ft_log_file = self._ft_log_path.open("w", newline="", encoding="utf-8")
            self._ft_log_writer = csv.writer(self._ft_log_file)
            self._ft_log_writer.writerow(
                [
                    "timestamp_utc",
                    "timestamp_unix_s",
                    "left_fx",
                    "left_fy",
                    "left_fz",
                    "left_tx",
                    "left_ty",
                    "left_tz",
                    "right_fx",
                    "right_fy",
                    "right_fz",
                    "right_tx",
                    "right_ty",
                    "right_tz",
                    "max_ft_force",
                ]
            )
            self._ft_log_file.flush()
            print(f"[INFO] Logging FT data to {self._ft_log_path.resolve()}", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to open FT log CSV: {exc}", flush=True)
            self._ft_log_file = None
            self._ft_log_writer = None
            self._ft_log_path = None

    def _close_ft_log(self) -> None:
        if self._ft_log_file is None:
            self._ft_log_writer = None
            self._ft_log_path = None
            return
        try:
            self._ft_log_file.flush()
            self._ft_log_file.close()
        except Exception:
            pass
        self._ft_log_file = None
        self._ft_log_writer = None
        self._ft_log_path = None

    def _setup_live_ft_plot(self) -> None:
        raw_axes = LIVE_FT_PLOT_AXES
        if raw_axes is None:
            raw_axes = LIVE_FT_PLOT_AXIS

        if raw_axes is None:
            return

        if isinstance(raw_axes, str):
            configured_axes = [raw_axes]
        elif isinstance(raw_axes, Sequence):
            configured_axes = [str(axis) for axis in raw_axes]
        else:
            print(
                "[WARN] LIVE_FT_PLOT_AXES must be a string or sequence of strings.",
                flush=True,
            )
            return

        axes: list[str] = []
        invalid_axes: list[str] = []
        seen: set[str] = set()
        for axis in configured_axes:
            normalized = str(axis).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if normalized not in FT_AXIS_TO_INDEX:
                invalid_axes.append(normalized)
                continue
            axes.append(normalized)

        if invalid_axes:
            print(
                f"[WARN] Ignoring invalid LIVE_FT plot axis values: {invalid_axes}. "
                f"Valid choices are {tuple(FT_AXIS_TO_INDEX.keys())}.",
                flush=True,
            )

        if not axes:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[WARN] Could not import matplotlib for live FT plot: {exc}", flush=True)
            return

        try:
            plt.ion()
            fig, axs = plt.subplots(len(axes), 1, num="Live FT", sharex=True)
            ax_list = list(np.atleast_1d(axs).reshape(-1))
            axis_to_ax: dict[str, Any] = {}
            axis_to_lines: dict[str, tuple[Any, Any]] = {}
            for axis, ax in zip(axes, ax_list):
                unit = "N" if axis.startswith("f") else "N*m"
                (left_line,) = ax.plot([], [], label=f"left_{axis}", linewidth=1.2)
                (right_line,) = ax.plot([], [], label=f"right_{axis}", linewidth=1.2)
                ax.set_ylabel(f"{axis} [{unit}]")
                if axis.startswith("f"):
                    ax.set_ylim(-float(LIVE_FT_FORCE_Y_LIMIT), float(LIVE_FT_FORCE_Y_LIMIT))
                elif axis.startswith("t"):
                    ax.set_ylim(-float(LIVE_FT_TORQUE_Y_LIMIT), float(LIVE_FT_TORQUE_Y_LIMIT))
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                axis_to_ax[axis] = ax
                axis_to_lines[axis] = (left_line, right_line)
            if ax_list:
                ax_list[-1].set_xlabel("time [s] from plot start")
            fig.tight_layout()
        except Exception as exc:
            print(f"[WARN] Failed to initialize live FT plot window: {exc}", flush=True)
            return

        self._live_plot_enabled = True
        self._live_plot_axes = tuple(axes)
        self._live_plot_axis_indices = {axis: int(FT_AXIS_TO_INDEX[axis]) for axis in self._live_plot_axes}
        self._live_plot_start_s = None
        self._live_plot_last_draw_s = 0.0
        self._live_plot_times.clear()
        self._live_plot_left_values = {axis: deque() for axis in self._live_plot_axes}
        self._live_plot_right_values = {axis: deque() for axis in self._live_plot_axes}
        self._live_plot_plt = plt
        self._live_plot_fig = fig
        self._live_plot_axes_by_name = axis_to_ax
        self._live_plot_lines = axis_to_lines
        print(f"[INFO] Live FT plot enabled for axes: {list(self._live_plot_axes)}", flush=True)

    def _teardown_live_ft_plot(self) -> None:
        if self._live_plot_plt is not None and self._live_plot_fig is not None:
            try:
                self._live_plot_plt.close(self._live_plot_fig)
            except Exception:
                pass
        self._live_plot_enabled = False
        self._live_plot_axes = tuple()
        self._live_plot_axis_indices = {}
        self._live_plot_start_s = None
        self._live_plot_last_draw_s = 0.0
        self._live_plot_times.clear()
        self._live_plot_left_values = {}
        self._live_plot_right_values = {}
        self._live_plot_plt = None
        self._live_plot_fig = None
        self._live_plot_axes_by_name = {}
        self._live_plot_lines = {}

    def _update_live_ft_plot(
        self,
        left_wrench: Optional[np.ndarray],
        right_wrench: Optional[np.ndarray],
    ) -> None:
        if not self._live_plot_enabled:
            return
        if not self._live_plot_axes:
            return
        if self._live_plot_plt is None or self._live_plot_fig is None:
            return
        if not self._live_plot_axes_by_name or not self._live_plot_lines:
            return

        now_s = time.monotonic()
        if self._live_plot_start_s is None:
            self._live_plot_start_s = now_s
        t_rel = float(now_s - self._live_plot_start_s)

        def _extract_axis_value(wrench: Optional[np.ndarray], axis_index: int) -> float:
            if wrench is None:
                return float("nan")
            arr = np.asarray(wrench, dtype=np.float64).reshape(-1)
            if arr.size <= axis_index:
                return float("nan")
            value = float(arr[axis_index])
            return value if math.isfinite(value) else float("nan")

        self._live_plot_times.append(t_rel)
        for axis in self._live_plot_axes:
            axis_index = self._live_plot_axis_indices.get(axis)
            if axis_index is None:
                continue
            self._live_plot_left_values[axis].append(_extract_axis_value(left_wrench, axis_index))
            self._live_plot_right_values[axis].append(_extract_axis_value(right_wrench, axis_index))

        window_s = float(LIVE_FT_PLOT_WINDOW_SECONDS)
        if window_s > 0.0:
            while self._live_plot_times and (t_rel - self._live_plot_times[0]) > window_s:
                self._live_plot_times.popleft()
                for axis in self._live_plot_axes:
                    if self._live_plot_left_values[axis]:
                        self._live_plot_left_values[axis].popleft()
                    if self._live_plot_right_values[axis]:
                        self._live_plot_right_values[axis].popleft()

        update_hz = float(max(LIVE_FT_PLOT_UPDATE_HZ, 0.5))
        min_period = 1.0 / update_hz
        if (now_s - self._live_plot_last_draw_s) < min_period:
            return
        self._live_plot_last_draw_s = now_s

        try:
            times = list(self._live_plot_times)
            for axis in self._live_plot_axes:
                left_line, right_line = self._live_plot_lines[axis]
                ax_obj = self._live_plot_axes_by_name[axis]
                left_line.set_data(times, list(self._live_plot_left_values[axis]))
                right_line.set_data(times, list(self._live_plot_right_values[axis]))
                ax_obj.relim()
                ax_obj.autoscale_view()
                if axis.startswith("f"):
                    ax_obj.set_ylim(-float(LIVE_FT_FORCE_Y_LIMIT), float(LIVE_FT_FORCE_Y_LIMIT))
                elif axis.startswith("t"):
                    ax_obj.set_ylim(-float(LIVE_FT_TORQUE_Y_LIMIT), float(LIVE_FT_TORQUE_Y_LIMIT))
                if window_s > 0.0:
                    xmin = max(0.0, t_rel - window_s)
                    xmax = max(t_rel, xmin + 1e-3)
                    ax_obj.set_xlim(xmin, xmax)
            self._live_plot_fig.canvas.draw_idle()
            self._live_plot_plt.pause(0.001)
        except Exception as exc:
            print(f"[WARN] Live FT plot update failed; disabling plot window: {exc}", flush=True)
            self._teardown_live_ft_plot()

    def _log_ft_row(
        self,
        left_wrench: Optional[np.ndarray],
        right_wrench: Optional[np.ndarray],
        max_force: Optional[float],
    ) -> None:
        if self._ft_log_writer is None or self._ft_log_file is None:
            return
        now = datetime.now(timezone.utc)
        left_w = None if left_wrench is None else np.asarray(left_wrench, dtype=np.float64).reshape(-1)
        right_w = None if right_wrench is None else np.asarray(right_wrench, dtype=np.float64).reshape(-1)
        try:
            self._ft_log_writer.writerow(
                [
                    now.isoformat(),
                    f"{now.timestamp():.6f}",
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 1 else left_w[0]),
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 2 else left_w[1]),
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 3 else left_w[2]),
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 4 else left_w[3]),
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 5 else left_w[4]),
                    _fmt_optional_scalar(None if left_w is None or left_w.size < 6 else left_w[5]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 1 else right_w[0]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 2 else right_w[1]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 3 else right_w[2]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 4 else right_w[3]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 5 else right_w[4]),
                    _fmt_optional_scalar(None if right_w is None or right_w.size < 6 else right_w[5]),
                    _fmt_optional_scalar(max_force),
                ]
            )
            self._ft_log_file.flush()
        except Exception:
            pass

    def _setup_keyboard_debug(self) -> None:
        try:
            app_window = omni_appwindow.get_default_app_window()
            if app_window is None:
                print("[WARN] No Isaac app window available for keyboard input.", flush=True)
                return
            self._keyboard = app_window.get_keyboard()
            if self._keyboard is None:
                print("[WARN] Isaac app window keyboard handle is unavailable.", flush=True)
                return
            self._input_iface = carb.input.acquire_input_interface()
            self._keyboard_sub = self._input_iface.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            print("[INFO] Keyboard debug subscription installed. Focus the Isaac Sim viewport/window to capture keys.", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to subscribe to keyboard events: {exc}", flush=True)
            self._keyboard = None
            self._keyboard_sub = None
            self._input_iface = None

    def _teardown_keyboard_debug(self) -> None:
        if self._input_iface is not None and self._keyboard is not None and self._keyboard_sub is not None:
            try:
                self._input_iface.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
        self._keyboard = None
        self._keyboard_sub = None
        self._input_iface = None

    def _on_keyboard_event(self, event: Any, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        try:
            if int(event.type) == int(carb.input.KeyboardEventType.KEY_PRESS):
                key_name = getattr(event.input, "name", None)
                if callable(key_name):
                    key_name = key_name()
                if not key_name:
                    key_name = str(event.input)
                normalized_key = str(key_name).upper()
                key_id = normalized_key
                for delimiter in (".", ":", " "):
                    if delimiter in key_id:
                        key_id = key_id.split(delimiter)[-1]

                if key_id == "T":
                    self._queue_keyboard_target_from_prim()
                elif key_id in ("UP", "UP_ARROW"):
                    self._step_keyboard_position(np.asarray([0.0, 0.0, 1.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id in ("DOWN", "DOWN_ARROW"):
                    self._step_keyboard_position(np.asarray([0.0, 0.0, -1.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id in ("LEFT", "LEFT_ARROW"):
                    self._step_keyboard_position(np.asarray([0.0, 1.0, 0.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id in ("RIGHT", "RIGHT_ARROW"):
                    self._step_keyboard_position(np.asarray([0.0, -1.0, 0.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id in KEYBOARD_FORWARD_KEYS:
                    self._step_keyboard_position(np.asarray([1.0, 0.0, 0.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id in KEYBOARD_BACKWARD_KEYS:
                    self._step_keyboard_position(np.asarray([-1.0, 0.0, 0.0], dtype=np.float64) * KEYBOARD_POSITION_STEP_M)
                elif key_id == "Q":
                    self._step_keyboard_orientation(np.asarray([1.0, 0.0, 0.0], dtype=np.float64), KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "W":
                    self._step_keyboard_orientation(np.asarray([1.0, 0.0, 0.0], dtype=np.float64), -KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "A":
                    self._step_keyboard_orientation(np.asarray([0.0, 1.0, 0.0], dtype=np.float64), KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "S":
                    self._step_keyboard_orientation(np.asarray([0.0, 1.0, 0.0], dtype=np.float64), -KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "Z":
                    self._step_keyboard_orientation(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "X":
                    self._step_keyboard_orientation(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), -KEYBOARD_ROTATION_STEP_RAD)
                elif key_id == "O":
                    self._step_gripper_target(KEYBOARD_GRIPPER_COARSE_STEP)
                elif key_id == "P":
                    self._step_gripper_target(-KEYBOARD_GRIPPER_COARSE_STEP)
                elif key_id in ("MINUS", "-"):
                    self._step_gripper_target(KEYBOARD_GRIPPER_STEP)
                elif key_id in ("EQUAL", "="):
                    self._step_gripper_target(-KEYBOARD_GRIPPER_STEP)
        except Exception as exc:
            print(f"[WARN] Keyboard event handling failed: {exc}", flush=True)
        return True

    def _queue_keyboard_target_from_prim(self) -> None:
        target_position, _ = _get_prim_world_pose(KEYBOARD_TARGET_PRIM_PATH)
        if target_position is None:
            print(
                f"[WARN] Could not read keyboard target position from '{KEYBOARD_TARGET_PRIM_PATH}'.",
                flush=True,
            )
            return
        self._keyboard_target_position = np.asarray(target_position, dtype=np.float64)
        print(
            f"[INFO] Queued keyboard target move to {KEYBOARD_TARGET_PRIM_PATH} at "
            f"{_fmt_seq(self._keyboard_target_position)}",
            flush=True,
        )

    def _step_keyboard_position(self, delta: np.ndarray) -> None:
        base_position = self._keyboard_target_position
        if base_position is None:
            if self._home_cmd_position is not None:
                base_position = np.asarray(self._home_cmd_position, dtype=np.float64)
            else:
                current_pose = self._get_home_pose_from_reference_prim()
                if current_pose is not None:
                    base_position = np.asarray(current_pose[0], dtype=np.float64)
        if base_position is None:
            print("[WARN] Could not resolve a base EEF position for keyboard motion.", flush=True)
            return

        self._keyboard_target_position = np.asarray(base_position, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
        print(
            "[INFO] Keyboard EEF position updated to "
            f"{_fmt_seq(self._keyboard_target_position)}",
            flush=True,
        )

    def _step_keyboard_orientation(self, axis: np.ndarray, angle_rad: float) -> None:
        base_orientation = self._keyboard_target_orientation
        if base_orientation is None:
            if self._home_cmd_orientation is not None:
                base_orientation = np.asarray(self._home_cmd_orientation, dtype=np.float64)
            else:
                current_pose = self._get_home_pose_from_reference_prim()
                if current_pose is not None:
                    base_orientation = np.asarray(current_pose[1], dtype=np.float64)
        if base_orientation is None:
            print("[WARN] Could not resolve a base EEF orientation for keyboard rotation.", flush=True)
            return

        delta_q = _quat_from_axis_angle_wxyz(axis, angle_rad)
        self._keyboard_target_orientation = _quat_normalize_wxyz(
            _quat_mul_wxyz(np.asarray(base_orientation, dtype=np.float64), delta_q)
        )
        print(
            "[INFO] Keyboard EEF orientation updated to "
            f"{_fmt_seq(self._keyboard_target_orientation)}",
            flush=True,
        )

    def _step_gripper_target(self, delta: float) -> None:
        base_width = self._gripper_target_width
        if base_width is None:
            current_width = self._read_current_gripper_width()
            base_width = current_width if current_width is not None else float(GRIPPER_MAX_POS)
        self._gripper_target_width = float(
            np.clip(float(base_width) + float(delta), GRIPPER_MIN_POS, GRIPPER_MAX_POS)
        )
        print(
            f"[INFO] Gripper target width set to {self._gripper_target_width:.5f} "
            f"(range {GRIPPER_MIN_POS:.5f}..{GRIPPER_MAX_POS:.5f})",
            flush=True,
        )

    def _stop_runtime_after_timeline_end(self) -> None:
        global _RUNTIME
        if self._stopped:
            return
        self.shutdown()
        if _RUNTIME is self:
            _RUNTIME = None
        print("[INFO] Timeline ended. Teleop runtime stopped. Run start() to run again.", flush=True)

    def _on_timeline_event(self, event: Any) -> None:
        event_type = int(event.type)
        stop_like_types = {int(omni.timeline.TimelineEventType.STOP)}
        pause_event = getattr(omni.timeline.TimelineEventType, "PAUSE", None)
        if pause_event is not None:
            stop_like_types.add(int(pause_event))
        if event_type in stop_like_types:
            self._stop_runtime_after_timeline_end()
            return
        if event_type == int(omni.timeline.TimelineEventType.PLAY):
            self._reset_cycle_state()
            self._reset_relative_reference_state()

    def _try_initialize_robot(self) -> bool:
        if self._fr3 is None or self._controller is None:
            return False
        try:
            self._fr3.initialize()
            self._articulation_controller = self._fr3.get_articulation_controller()
            self._controller.reset()
            self._robot_ready = True
            self._warned_articulation_init = False
            return True
        except Exception as exc:
            # In attach mode, physics/simulation view can be temporarily unavailable
            # around timeline transitions. Keep retrying on future steps.
            self._robot_ready = False
            self._articulation_controller = None
            if not self._warned_articulation_init:
                print(
                    "[WARN] Articulation view not ready yet; will retry initialization on next physics steps. "
                    f"Last error: {exc}",
                    flush=True,
                )
                self._warned_articulation_init = True
            return False

    def _build_home_joint_positions(self) -> Optional[np.ndarray]:
        if self._fr3 is None:
            return None
        if self._home_joint_positions_cache is not None:
            return self._home_joint_positions_cache

        try:
            current = np.asarray(self._fr3.get_joint_positions(), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if current.size == 0:
            return None

        dof_names = getattr(self._fr3, "dof_names", None)
        if dof_names is None:
            vals = list(FR3_HOME_JOINT_TARGETS.values())
            if len(vals) != int(current.size):
                self._home_joint_targets_available = False
                print("[WARN] Could not resolve articulation dof names; skipping auto-homing.", flush=True)
                return None
            target = np.asarray(vals, dtype=np.float64)
            self._home_joint_positions_cache = target
            return target

        target = current.copy()
        for i, name in enumerate(dof_names):
            if name in FR3_HOME_JOINT_TARGETS:
                target[i] = float(FR3_HOME_JOINT_TARGETS[name])
        self._home_joint_positions_cache = target
        return target

    def _apply_home_joint_positions(self) -> bool:
        if self._fr3 is None:
            return False
        target = self._build_home_joint_positions()
        if target is None:
            return False
        try:
            self._fr3.set_joint_positions(target)
            return True
        except Exception as exc:
            if self._home_joint_targets_available:
                print(f"[WARN] Failed to apply auto-home joint target: {exc}", flush=True)
                self._home_joint_targets_available = False
            return False

    def _run_startup_homing(self, dt: float) -> bool:
        if self._homing_done:
            return True
        if not self._home_joint_targets_available:
            self._homing_done = True
            return True

        if self._homing_elapsed_s <= 0.0:
            print(
                "[INFO] Auto-homing FR3 to startup joint target and waiting "
                f"{FR3_HOME_SETTLE_SECONDS:.1f}s before latching initial EEF pose.",
                flush=True,
            )

        _ = self._apply_home_joint_positions()
        self._homing_elapsed_s += float(max(dt, 0.0))
        if self._homing_elapsed_s < float(FR3_HOME_SETTLE_SECONDS):
            return False

        self._homing_done = True
        self._input_ref_position = None
        self._input_ref_orientation = None
        self._home_cmd_position = None
        self._home_cmd_orientation = None
        self._printed_initial_pose_snapshot = False
        print(
            "[INFO] Auto-homing complete. The first valid teleop pose will be treated as the input reference, "
            "and it will map to the FR3 home EEF pose reached by the fixed startup joint target.",
            flush=True,
        )
        return True

    def _get_home_pose_from_reference_prim(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        try:
            for prim_path in [EE_HOME_REFERENCE_PRIM_PATH] + list(EE_HOME_REFERENCE_PRIM_FALLBACK_PATHS):
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                world_mtx = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = world_mtx.ExtractTranslation()
                q = world_mtx.ExtractRotationQuat()
                pos = np.asarray([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
                ori = np.asarray(
                    [float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2])],
                    dtype=np.float64,
                )
                if prim_path != EE_HOME_REFERENCE_PRIM_PATH:
                    print(f"[WARN] Using fallback home reference prim '{prim_path}'.", flush=True)
                return pos, _quat_normalize_wxyz(ori)
            return None
        except Exception:
            return None

    def _ensure_home_command_pose(self) -> bool:
        if self._home_cmd_position is not None and self._home_cmd_orientation is not None:
            return True
        home_pose = self._get_home_pose_from_reference_prim()
        if home_pose is None:
            if not self._warned_waiting_home_pose:
                print(
                    f"[WARN] Waiting for FR3 EEF world pose from '{EE_HOME_REFERENCE_PRIM_PATH}' "
                    "before applying teleop targets.",
                    flush=True,
                )
                self._warned_waiting_home_pose = True
            return False
        pos, ori = home_pose
        self._home_cmd_position = np.asarray(pos, dtype=np.float64)
        self._home_cmd_orientation = _quat_normalize_wxyz(ori)
        if self._keyboard_target_orientation is None:
            self._keyboard_target_orientation = np.asarray(self._home_cmd_orientation, dtype=np.float64)
        self._warned_waiting_home_pose = False
        return True

    def _to_relative_command(
        self, input_position: np.ndarray, input_orientation: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if (
            self._input_ref_position is None
            or self._input_ref_orientation is None
            or self._home_cmd_position is None
            or self._home_cmd_orientation is None
        ):
            return None

        # Keep quaternion sign consistent to avoid discontinuities around +/-q equivalence.
        in_ori = _quat_normalize_wxyz(input_orientation)
        if float(np.dot(in_ori, self._input_ref_orientation)) < 0.0:
            in_ori = -in_ori

        delta_pos_world = np.asarray(input_position, dtype=np.float64) - self._input_ref_position
        if TRANSLATION_MAPPING_MODE == "local_delta":
            # Optional test mode: express translation in leader-reference local frame,
            # then re-express that local delta in home frame.
            r_ref = _quat_to_rotmat_wxyz(self._input_ref_orientation)
            delta_local = r_ref.T @ delta_pos_world
            r_home = _quat_to_rotmat_wxyz(self._home_cmd_orientation)
            cmd_pos = self._home_cmd_position + float(POSITION_DELTA_SCALE) * (r_home @ delta_local)
        else:
            # Default mode: directly apply world-frame delta from first input.
            cmd_pos = self._home_cmd_position + float(POSITION_DELTA_SCALE) * delta_pos_world

        if ORIENTATION_MAPPING_MODE == "conjref_current__home_delta":
            delta_ori = _quat_mul_wxyz(_quat_conj_wxyz(self._input_ref_orientation), in_ori)
            cmd_ori = _quat_normalize_wxyz(_quat_mul_wxyz(self._home_cmd_orientation, delta_ori))
        else:
            delta_ori = _quat_mul_wxyz(in_ori, _quat_conj_wxyz(self._input_ref_orientation))
            cmd_ori = _quat_normalize_wxyz(_quat_mul_wxyz(delta_ori, self._home_cmd_orientation))

        return cmd_pos, cmd_ori

    def _ensure_initial_relative_references(
        self, input_position: np.ndarray, input_orientation: np.ndarray
    ) -> bool:
        if (
            self._input_ref_position is None
            or self._input_ref_orientation is None
            or self._home_cmd_position is None
            or self._home_cmd_orientation is None
        ):
            # Latch the first valid incoming teleop pose as the input reference.
            # That first input maps to the FR3 EEF pose produced by the fixed startup joint target.
            if not self._ensure_home_command_pose():
                return False
            self._input_ref_position = np.asarray(input_position, dtype=np.float64)
            self._input_ref_orientation = _quat_normalize_wxyz(input_orientation)

        if not self._printed_initial_pose_snapshot:
            print(
                "[INFO] Initial pose references latched: "
                f"initial_command_pos={_fmt_seq(self._input_ref_position)}, "
                f"initial_command_ori_wxyz={_fmt_seq(self._input_ref_orientation)}, "
                f"initial_fr3_eef_pos={_fmt_seq(self._home_cmd_position)}, "
                f"initial_fr3_eef_ori_wxyz={_fmt_seq(self._home_cmd_orientation)}",
                flush=True,
            )
            self._printed_initial_pose_snapshot = True

        return True

    def _reanchor_relative_command(
        self,
        input_position: np.ndarray,
        input_orientation: np.ndarray,
        command_position: np.ndarray,
        command_orientation: Optional[np.ndarray],
    ) -> None:
        self._input_ref_position = np.asarray(input_position, dtype=np.float64)
        self._input_ref_orientation = _quat_normalize_wxyz(np.asarray(input_orientation, dtype=np.float64))
        self._home_cmd_position = np.asarray(command_position, dtype=np.float64)
        if command_orientation is None:
            current_home = self._home_cmd_orientation
            if current_home is None:
                current_pose = self._get_home_pose_from_reference_prim()
                command_orientation = current_pose[1] if current_pose is not None else np.asarray(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
            else:
                command_orientation = current_home
        self._home_cmd_orientation = _quat_normalize_wxyz(np.asarray(command_orientation, dtype=np.float64))
        self._keyboard_target_orientation = np.asarray(self._home_cmd_orientation, dtype=np.float64)
        self._printed_initial_pose_snapshot = False
        self._warned_waiting_home_pose = False
        print(
            "[INFO] Relative teleop reference re-anchored: "
            f"input_ref_pos={_fmt_seq(self._input_ref_position)}, "
            f"home_cmd_pos={_fmt_seq(self._home_cmd_position)}",
            flush=True,
        )

    def _read_target_pose_from_omnigraph(self) -> Optional[Tuple[list[float], list[float]]]:
        # OmniGraph data read happens here each step from the ROS2 Subscribe Pose node outputs.
        raw_position = None
        for attr_path in self._position_attr_paths:
            raw_position = _read_og_attr(attr_path)
            if raw_position is not None:
                break

        raw_orientation = None
        for attr_path in self._orientation_attr_paths:
            raw_orientation = _read_og_attr(attr_path)
            if raw_orientation is not None:
                break

        position = _as_xyz(raw_position)
        orientation = _as_wxyz(raw_orientation)
        if position is not None and orientation is not None:
            return position, orientation

        # Leaf fallback for structured vector/quaternion outputs that may return None at parent path.
        for pos_base in self._position_attr_paths:
            px = _read_og_leaf_scalar(pos_base, "x")
            py = _read_og_leaf_scalar(pos_base, "y")
            pz = _read_og_leaf_scalar(pos_base, "z")
            if None not in (px, py, pz):
                position = [float(px), float(py), float(pz)]
                break
        else:
            position = None

        for ori_base in self._orientation_attr_paths:
            ox = _read_og_leaf_scalar(ori_base, "x")
            oy = _read_og_leaf_scalar(ori_base, "y")
            oz = _read_og_leaf_scalar(ori_base, "z")
            ow = _read_og_leaf_scalar(ori_base, "w")
            if None not in (ox, oy, oz, ow):
                orientation = [float(ow), float(ox), float(oy), float(oz)]
                break
        else:
            orientation = None

        if position is not None and orientation is not None:
            return position, orientation

        for attr_path in self._pose_attr_paths:
            raw_pose = _read_og_attr(attr_path)
            pose = _extract_pose_from_struct(raw_pose)
            if pose is not None:
                return pose
        return None

    def _read_gripper_width_from_omnigraph(self) -> Optional[float]:
        for attr_path in self._gripper_attr_paths:
            raw_val = _read_og_attr(attr_path)
            width = _extract_gripper_width(raw_val)
            if width is not None:
                return width
        return None

    def _get_gripper_joint_indices(self) -> Optional[Tuple[int, int]]:
        if self._gripper_joint_indices is not None:
            return self._gripper_joint_indices
        if self._fr3 is None:
            return None
        dof_names = getattr(self._fr3, "dof_names", None)
        if not isinstance(dof_names, Sequence):
            return None
        try:
            idx1 = list(dof_names).index(GRIPPER_JOINT_NAMES[0])
            idx2 = list(dof_names).index(GRIPPER_JOINT_NAMES[1])
            self._gripper_joint_indices = (int(idx1), int(idx2))
            return self._gripper_joint_indices
        except Exception:
            if not self._warned_gripper_joint_missing:
                print(
                    f"[WARN] Gripper joints {GRIPPER_JOINT_NAMES} not found in articulation DOFs.",
                    flush=True,
                )
                self._warned_gripper_joint_missing = True
            return None

    def _apply_gripper_width(self, width: float) -> None:
        if self._fr3 is None:
            return
        indices = self._get_gripper_joint_indices()
        if indices is None:
            return
        target = float(np.clip(width, GRIPPER_MIN_POS, GRIPPER_MAX_POS))
        try:
            joint_positions = np.asarray(self._fr3.get_joint_positions(), dtype=np.float64).reshape(-1)
            if joint_positions.size <= max(indices):
                return
            joint_positions[indices[0]] = target
            joint_positions[indices[1]] = target
            self._fr3.set_joint_positions(joint_positions)
        except Exception as exc:
            print(f"[WARN] Failed to apply gripper command {target:.3f}: {exc}", flush=True)

    @staticmethod
    def _force_value_to_xyz(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if all(hasattr(value, a) for a in ("x", "y", "z")):
            try:
                vec = np.asarray([float(value.x), float(value.y), float(value.z)], dtype=np.float64)
                return vec if np.all(np.isfinite(vec)) else None
            except Exception:
                return None
        for force_attr in ("force", "linear", "linear_force"):
            force_value = getattr(value, force_attr, None)
            if force_value is not None:
                vec = FrankaTeleopAttachRuntime._force_value_to_xyz(force_value)
                if vec is not None:
                    return vec
        for getter_name in ("GetForce", "GetLinear"):
            getter = getattr(value, getter_name, None)
            if callable(getter):
                try:
                    vec = FrankaTeleopAttachRuntime._force_value_to_xyz(getter())
                except Exception:
                    vec = None
                if vec is not None:
                    return vec
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
            try:
                arr = np.asarray([float(v) for v in value], dtype=np.float64).reshape(-1)
                vec = arr[:3]
                return vec if vec.size == 3 and np.all(np.isfinite(vec)) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _force_value_to_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        vec = FrankaTeleopAttachRuntime._force_value_to_xyz(value)
        if vec is not None:
            n = float(np.linalg.norm(vec))
            return n if math.isfinite(n) else None
        try:
            scalar = float(value)
            return abs(scalar) if math.isfinite(scalar) else None
        except Exception:
            pass
        for force_attr in ("force", "linear", "linear_force"):
            force_value = getattr(value, force_attr, None)
            if force_value is not None:
                scalar = FrankaTeleopAttachRuntime._force_value_to_scalar(force_value)
                if scalar is not None:
                    return scalar
        for getter_name in ("GetForce", "GetLinear"):
            getter = getattr(value, getter_name, None)
            if callable(getter):
                try:
                    scalar = FrankaTeleopAttachRuntime._force_value_to_scalar(getter())
                except Exception:
                    scalar = None
                if scalar is not None:
                    return scalar
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 1:
            try:
                arr = np.asarray([float(v) for v in value], dtype=np.float64).reshape(-1)
                if arr.size >= 6:
                    arr = arr[:3]
                n = float(np.linalg.norm(arr))
                return n if math.isfinite(n) else None
            except Exception:
                return None
        return None

    def _read_ft_force_vector(self, prim_path: str) -> Optional[np.ndarray]:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        selected = self._ft_selected_attr.get(prim_path)
        attr_names = [selected] if selected else []
        attr_names.extend([name for name in FT_FORCE_ATTR_CANDIDATES if name != selected])
        for attr_name in attr_names:
            if not attr_name:
                continue
            try:
                attr = prim.GetAttribute(attr_name)
                if not attr or not attr.IsValid():
                    continue
                force_vec = self._force_value_to_xyz(attr.Get())
                if force_vec is None:
                    continue
                self._ft_selected_attr[prim_path] = attr_name
                return force_vec
            except Exception:
                continue
        return None

    def _read_ft_force_scalar(self, prim_path: str) -> Optional[float]:
        force_vec = self._read_ft_force_vector(prim_path)
        if force_vec is not None:
            force_val = float(np.linalg.norm(force_vec))
            return force_val if math.isfinite(force_val) else None
        return None

    def _resolve_ft_link_ids(self) -> Optional[Tuple[int, int]]:
        if self._ft_link_ids is not None:
            return self._ft_link_ids
        if self._fr3 is None:
            return None

        articulation_view = getattr(self._fr3, "_articulation_view", None)
        if articulation_view is not None:
            body_names = getattr(articulation_view, "body_names", None)
            if isinstance(body_names, Sequence):
                body_names = list(body_names)
                try:
                    left_name = next(name for name in LEFT_FT_JOINT_NAME_CANDIDATES if name in body_names)
                    right_name = next(name for name in RIGHT_FT_JOINT_NAME_CANDIDATES if name in body_names)
                    left_id = int(articulation_view.get_body_index(left_name))
                    right_id = int(articulation_view.get_body_index(right_name))
                    self._ft_link_ids = (left_id, right_id)
                    self._ft_link_names = (str(left_name), str(right_name))
                    if not self._printed_ft_joint_resolution:
                        print(
                            "[INFO] Resolved FT link ids for measured articulation forces: "
                            f"left_joint_index={left_id} ({left_name}), "
                            f"right_joint_index={right_id} ({right_name})",
                            flush=True,
                        )
                        self._printed_ft_joint_resolution = True
                    return self._ft_link_ids
                except StopIteration:
                    if not self._warned_ft_joint_force_missing:
                        print(
                            "[WARN] FT body-name resolution failed. "
                            f"Expected left in {LEFT_FT_JOINT_NAME_CANDIDATES}, "
                            f"right in {RIGHT_FT_JOINT_NAME_CANDIDATES}. "
                            f"Available articulation body names: {body_names}",
                            flush=True,
                        )
                    pass
                except Exception:
                    pass

        if not self._warned_ft_joint_force_missing:
            print(
                "[WARN] Could not resolve FT link ids for measured articulation forces; "
                "falling back to configured prim paths.",
                flush=True,
            )
            self._warned_ft_joint_force_missing = True
        return None

    def _read_measured_link_wrench(self, link_id: int) -> Optional[np.ndarray]:
        if self._fr3 is None:
            return None
        try:
            measured = np.asarray(self._fr3.get_measured_joint_forces(), dtype=np.float64)
        except Exception:
            return None
        if measured.size == 0:
            return None

        if measured.ndim == 3:
            measured = measured[0]
        if measured.ndim != 2:
            return None
        if link_id < 0:
            return None
        if link_id >= measured.shape[0]:
            return None

        wrench = np.asarray(measured[link_id], dtype=np.float64).reshape(-1)
        if wrench.size < 6:
            return None
        return wrench[:6] if np.all(np.isfinite(wrench[:6])) else None

    def _read_gripper_contact_forces(
        self,
    ) -> Tuple[
        str,
        Optional[int],
        Optional[str],
        Optional[np.ndarray],
        Optional[np.ndarray],
        str,
        Optional[int],
        Optional[str],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
    ]:
        left_source = "none"
        left_joint_index = None
        left_joint_name = None
        left_wrench = None
        left_xyz = None
        right_source = "none"
        right_joint_index = None
        right_joint_name = None
        right_wrench = None
        right_xyz = None

        ft_link_ids = self._resolve_ft_link_ids()
        if ft_link_ids is not None:
            left_joint_index = int(ft_link_ids[0])
            right_joint_index = int(ft_link_ids[1])
            if self._ft_link_names is not None:
                left_joint_name, right_joint_name = self._ft_link_names

            left_wrench = self._read_measured_link_wrench(left_joint_index)
            if left_wrench is not None:
                left_xyz = np.asarray(left_wrench[:3], dtype=np.float64)
                left_source = "measured_articulation"

            right_wrench = self._read_measured_link_wrench(right_joint_index)
            if right_wrench is not None:
                right_xyz = np.asarray(right_wrench[:3], dtype=np.float64)
                right_source = "measured_articulation"

        if left_xyz is None:
            left_xyz = self._read_ft_force_vector(LEFT_FT_JOINT_PRIM_PATH)
            if left_xyz is not None:
                left_source = f"prim_attr:{self._ft_selected_attr.get(LEFT_FT_JOINT_PRIM_PATH, '')}"
        if right_xyz is None:
            right_xyz = self._read_ft_force_vector(RIGHT_FT_JOINT_PRIM_PATH)
            if right_xyz is not None:
                right_source = f"prim_attr:{self._ft_selected_attr.get(RIGHT_FT_JOINT_PRIM_PATH, '')}"
        vals = [
            float(np.linalg.norm(v))
            for v in (left_xyz, right_xyz)
            if v is not None and np.all(np.isfinite(v))
        ]
        if not vals:
            if not self._warned_ft_unavailable:
                print(
                    "[WARN] Could not read fingertip FT force values from configured prim paths.",
                    flush=True,
                )
                self._warned_ft_unavailable = True
            return (
                left_source,
                left_joint_index,
                left_joint_name,
                left_wrench,
                left_xyz,
                right_source,
                right_joint_index,
                right_joint_name,
                right_wrench,
                right_xyz,
                None,
            )
        self._warned_ft_unavailable = False
        return (
            left_source,
            left_joint_index,
            left_joint_name,
            left_wrench,
            left_xyz,
            right_source,
            right_joint_index,
            right_joint_name,
            right_wrench,
            right_xyz,
            max(vals),
        )

    def _read_current_gripper_width(self) -> Optional[float]:
        if self._fr3 is None:
            return None
        indices = self._get_gripper_joint_indices()
        if indices is None:
            return None
        try:
            joint_positions = np.asarray(self._fr3.get_joint_positions(), dtype=np.float64).reshape(-1)
            if joint_positions.size <= max(indices):
                return None
            return float(0.5 * (joint_positions[indices[0]] + joint_positions[indices[1]]))
        except Exception:
            return None

    def _update_gripper_smooth(self, dt: float) -> None:
        if self._gripper_target_width is None:
            return

        if self._gripper_applied_width is None:
            current_width = self._read_current_gripper_width()
            self._gripper_applied_width = (
                current_width if current_width is not None else self._gripper_target_width
            )

        ft_read = self._read_gripper_contact_forces()
        left_wrench = ft_read[3]
        right_wrench = ft_read[8]
        max_force = ft_read[10]
        self._log_ft_row(
            left_wrench,
            right_wrench,
            max_force,
        )
        self._update_live_ft_plot(left_wrench, right_wrench)

        opening_requested = self._gripper_target_width > (self._gripper_applied_width + 1e-6)

        if opening_requested:
            self._gripper_contact_latched = False

        max_step = float(GRIPPER_MAX_SPEED) * float(max(dt, 1e-6))
        error = self._gripper_target_width - self._gripper_applied_width
        step = float(np.clip(error, -max_step, max_step))
        self._gripper_applied_width += step
        self._apply_gripper_width(self._gripper_applied_width)

    def _on_update(self, event: Any) -> None:
        del event
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return
        self._latest_target_pose = self._read_target_pose_from_omnigraph()

    def _set_target_world_pose(self, position: Sequence[float], orientation_wxyz: Sequence[float]) -> None:
        if self._disable_target_updates or self._target is None:
            return
        # Isaac API variants:
        # - set_world_pose(position=..., orientation=...)
        # - set_world_poses(positions=..., orientations=...) (batched)
        try:
            self._target.set_world_pose(position=position, orientation=orientation_wxyz)
            return
        except Exception:
            pass
        try:
            self._target.set_world_pose(position, orientation_wxyz)
            return
        except Exception:
            pass
        try:
            positions = np.asarray([position], dtype=np.float32)
            orientations = np.asarray([orientation_wxyz], dtype=np.float32)
            self._target.set_world_poses(positions=positions, orientations=orientations)
            return
        except Exception:
            pass
        try:
            positions = np.asarray([position], dtype=np.float32)
            orientations = np.asarray([orientation_wxyz], dtype=np.float32)
            self._target.set_world_poses(positions, orientations)
        except Exception as exc:
            print(f"[WARN] Failed to update target pose visualization: {exc}", flush=True)
            self._disable_target_updates = True
            self._target = None

    def _on_physics_step(self, dt: float) -> None:
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        if self._controller is None or self._fr3 is None:
            return

        if self._reset_needed:
            # Never call world.reset() inside physics callback (can recurse/crash).
            # Reinitialize articulation/controller state in-place when timeline restarts.
            if not self._try_initialize_robot():
                return
            self._reset_needed = False

        if not self._robot_ready or self._articulation_controller is None:
            self._try_initialize_robot()
            return

        if not self._run_startup_homing(dt):
            return

        self._update_gripper_smooth(dt)

        keyboard_target = self._keyboard_target_position
        keyboard_orientation = self._keyboard_target_orientation
        if keyboard_target is not None:
            self._keyboard_target_position = None
            if not self._ensure_home_command_pose():
                return
            target_position_np = np.asarray(keyboard_target, dtype=np.float64)
            target_orientation_np = (
                np.asarray(keyboard_orientation, dtype=np.float64)
                if keyboard_orientation is not None
                else np.asarray(self._home_cmd_orientation, dtype=np.float64)
            )
            latest_pose = self._latest_target_pose
            if USE_RELATIVE_INPUT and latest_pose is not None:
                latest_input_position = np.asarray(latest_pose[0], dtype=np.float64)
                latest_input_orientation = _quat_normalize_wxyz(np.asarray(latest_pose[1], dtype=np.float64))
                self._reanchor_relative_command(
                    input_position=latest_input_position,
                    input_orientation=latest_input_orientation,
                    command_position=target_position_np,
                    command_orientation=target_orientation_np,
                )
        else:
            pose = self._latest_target_pose
            if pose is None:
                if not self._ensure_home_command_pose():
                    return
                self._missing_pose_steps += 1
                if not self._warned_waiting_for_pose:
                    print(
                        "[INFO] No /eef_pose input yet. Holding the startup home pose and accepting keyboard targets.",
                        flush=True,
                    )
                    self._warned_waiting_for_pose = True
                target_position_np = np.asarray(self._home_cmd_position, dtype=np.float64)
                target_orientation_np = (
                    np.asarray(keyboard_orientation, dtype=np.float64)
                    if keyboard_orientation is not None
                    else np.asarray(self._home_cmd_orientation, dtype=np.float64)
                )
            else:
                self._missing_pose_steps = 0

                target_position, target_orientation = pose
                target_position_np = np.asarray(target_position, dtype=np.float64)
                target_orientation_np = _quat_normalize_wxyz(np.asarray(target_orientation, dtype=np.float64))
                if USE_RELATIVE_INPUT:
                    if not self._ensure_initial_relative_references(target_position_np, target_orientation_np):
                        return
                    relative_cmd = self._to_relative_command(
                        target_position_np, target_orientation_np
                    )
                    if relative_cmd is None:
                        return
                    target_position_np, target_orientation_np = relative_cmd
                if keyboard_orientation is not None:
                    target_orientation_np = np.asarray(keyboard_orientation, dtype=np.float64)

        if keyboard_target is None or keyboard_orientation is not None:
            self._set_target_world_pose(target_position_np, target_orientation_np)
            target_ori_arg = target_orientation_np if SEND_ORIENTATION_TARGET else None
        else:
            vis_orientation = (
                keyboard_orientation
                if keyboard_orientation is not None
                else self._home_cmd_orientation
                if self._home_cmd_orientation is not None
                else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            )
            self._set_target_world_pose(target_position_np, vis_orientation)
            target_ori_arg = None

        actions = self._controller.forward(
            target_end_effector_position=target_position_np,
            target_end_effector_orientation=target_ori_arg,
        )
        self._articulation_controller.apply_action(actions)


_RUNTIME: Optional[FrankaTeleopAttachRuntime] = None


def debug_ft_prims() -> None:
    _ensure_isaac_imports()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] No USD stage is currently open in Isaac Sim.", flush=True)
        return

    for prim_path in (LEFT_FT_JOINT_PRIM_PATH, RIGHT_FT_JOINT_PRIM_PATH):
        print(f"[INFO] Inspecting FT prim: {prim_path}", flush=True)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            print("[WARN] Prim does not exist or is invalid.", flush=True)
            continue

        attr_names = sorted(attr.GetName() for attr in prim.GetAttributes())
        print(f"[INFO] Attribute count: {len(attr_names)}", flush=True)
        print(f"[INFO] Attributes: {attr_names}", flush=True)

        candidate_names = list(dict.fromkeys(FT_FORCE_ATTR_CANDIDATES + attr_names))
        for attr_name in candidate_names:
            if not attr_name:
                continue
            try:
                attr = prim.GetAttribute(attr_name)
                if not attr or not attr.IsValid():
                    continue
                raw_value = attr.Get()
                scalar = FrankaTeleopAttachRuntime._force_value_to_scalar(raw_value)
                print(
                    f"[INFO] {prim_path} :: {attr_name} -> raw={raw_value!r}, scalar={scalar}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[WARN] {prim_path} :: {attr_name} read failed: {exc}",
                    flush=True,
                )


def start() -> None:
    global _RUNTIME
    _ensure_isaac_imports()
    if _RUNTIME is not None:
        print("[WARN] Teleop runtime already started.", flush=True)
        return
    _RUNTIME = FrankaTeleopAttachRuntime()
    _RUNTIME.start()


if __name__ == "__main__":
    start()
