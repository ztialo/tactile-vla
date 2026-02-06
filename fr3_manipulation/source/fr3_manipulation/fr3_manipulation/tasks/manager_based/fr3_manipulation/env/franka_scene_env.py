"""
Single arm environment
"""

from typing import Sequence, Dict, List
import torch
import omni.usd
from pxr import Usd, UsdShade, Gf, Sdf
import numpy as np
import isaaclab.sim as sim_utils
import omni.kit.commands
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    matrix_from_euler,
    quat_from_euler_xyz,
)


class FrankaEnv(DirectRLEnv):
    """
    Single arm pick & place environment
    """

    def __init__(self, cfg: FrankaEnvCfg):
        super().__init__(cfg)
        self.stage = omni.usd.get_context().get_stage()
        self.robot = self.scene.articulations["franka"]
        # self.cube = self.scene.rigid_objects["dice_cube"]
        # self.desk_cam = self.scene.sensors["desk_cam"]
        # self.hand_cam = self.scene.sensors["hand_cam"]
        # self.obstacle_wall = self.scene.rigid_objects["obstacle_wall"]
        # self.obstacle_bool = False
        # self.torch_rng = torch.Generator(device=self.device)
        # self.torch_rng.seed()
        # self.target_goal = torch.zeros(
        #     (self.num_envs, 3), device=self.device, dtype=torch.float32
        # )
        # self._set_up_debug_vis()
        # self.is_locked = torch.zeros(
        #     (self.num_envs,), device=self.device, dtype=torch.bool
        # )
        # self._set_up_diff_ik_controller()
        # # pyroki planner
        # self.world_coll_list = []
        # self.planner = IKwCollision("piper", self.robot)
        # # Buffer
        # self._actions = torch.zeros(
        #     (self.num_envs, 7), device=self.device, dtype=torch.float32
        # )
        # self._actions_all = torch.zeros(
        #     (self.num_envs, 8), device=self.device, dtype=torch.float32
        # )
        # self.randomizer_prim_list = None

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # Reset the robot
        dof_pos = self.robot.data.default_joint_pos.clone()
        dof_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)


    def _set_up_diff_ik_controller(self) -> None:
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )
        self.arm_entity_cfg = SceneEntityCfg(
            "franka", joint_names=["joint[1-6]"], body_names=["link[6]"]
        )
        self.arm_entity_cfg.resolve(self.scene)

    def compute_ik(
        self, target_poses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the joint action from the target poses
        """
        # transform the ee pose from world to base frame
        ee_pose_local = target_poses["ee_pose"].clone().detach()
        ee_pose_local[:, 1] += 0.25  # y offset from robot initial pos
        # Update command
        ik_commands = ee_pose_local
        self.diff_ik_controller.set_command(ik_commands)

        # return ik_commands
        ee_pose_w = self.robot.data.body_pose_w[:, self.arm_entity_cfg.body_ids[0]]
        ee_jacobi_idx = (
            self.arm_entity_cfg.body_ids[0] - 1
        )  # minus 1 because the jacobian does not include the base
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, self.arm_entity_cfg.joint_ids
        ]
        # ee_pose_w = self.robot.data.body_pose_w[:, self.arm_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.arm_entity_cfg.joint_ids]
        # Compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        # Compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        # Update the ee goal frame
        self.ee_goal_pose = target_poses["ee_pose"].clone().detach()
        # Gripper value
        joint_pos_all = torch.zeros([self.num_envs, 7], device=self.device)
        joint_pos_all[:, :6] = joint_pos_des
        joint_pos_all[:, 6] = 1 - target_poses["gripper_val"]
        return joint_pos_all