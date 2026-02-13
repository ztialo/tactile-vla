import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Franka arm tactile data collection."
)
# parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--num_max_episodes", type=int, default=3000, help="Number of maximum episodes."
)
# Override config
parser.add_argument(
    "--show_gui", action="store_true", default=True, help="Whether to show the GUI."
)
parser.add_argument(
    "--obstacle", action="store_true", default=False, help="Whether to add obstacle."
)
parser.add_argument(
    "--control_type",
    type=str,
    default="teleop",
    help="Type of control: teleop, ik, or pyroki",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_dict = {
    # "device": "cuda",  # cpu contact is different with gpu; gpu contact is bad to small objects
    # "active_gpu": 1,  # gpu id for rendering
    "max_gpu_count": 1,
    "enable_cameras": True,
    # "headless": False,
    "experience": "",
    "livestream": 0,
    "verbose": False,
    "multi_gpu": True,
    "renderer": "RayTracedLighting",  # real-time
    # "renderer": "PathTracing",  # FIXME: Can not launch with PathTracing
    "samples_per_pixel_per_frame": 16,  # 8 is minimum
}
# if args_cli.real_time:
#     app_dict["renderer"] = "RayTracedLighting"
# Override some of the arguments from the cli
app_dict["device"] = args_cli.device
# app_dict["device"] = "cpu"  # Use CPU for inference
# app_dict["headless"] = not args_cli.show_gui
app_dict["headless"] = False

app_dict_bk = app_dict.copy()  # backup

# launch omniverse app
app_launcher = AppLauncher(app_dict)
simulation_app = app_launcher.app

"""Rest everything follows."""
import time
import cv2
import json
import torch
import yaml
import websocket
import msgpack
import numpy as np
import os
from typing import Dict

from .franka_scene_env import FrankaEnvCfg, FrankaEnv

def main():

    num_max_episodes = args_cli.num_max_episodes
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    control_type = args_cli.control_type
    obstacle = args_cli.obstacle
    # control_type = "ik"
    # control_type = "teleop"
    # control_type = "infer"
    if control_type == "teleop":
        # teleop control with starai arm using ik controller from ros node
        print("Using teleop control with StarAI arm")
    else:
        print("invalid control_type: ", control_type)
        raise ValueError(f"Invalid control type: {control_type}")

    cfg = FrankaEnvCfg()
    env = FrankaEnv(cfg)

    # if obstacle:
    #     env.set_obstacle()

    env.reset()
    count = 0

    action_queue = None
    while simulation_app.is_running():
        if control_type == "teleop":
            # read ee pose and gripper command from ros node
            # goal_pose = env.read_cmds_ros()
            # gello_qpos = env.compute_ik(goal_pose).squeeze(0).cpu().numpy()
            # action_queue = gello_qpos[None, :]
            pass
        # elif control_type == "ik":
        #     task_state = env.get_task_state()
        #     state_machine.update_sm(task_state)
        #     goal_pose = state_machine.get_goal_pose(task_state)
        #     gello_qpos = env.compute_ik(goal_pose).squeeze(0).cpu().numpy()
        #     action_queue = gello_qpos[None, :]
        else:
            raise ValueError(f"Invalid control type: {control_type}")

        # Loop over action queue
        # for gello_qpos in action_queue:
        #     obs, rew, done, truncated, extras = env.step(
        #         torch.from_numpy(gello_qpos[:7]).to(env.device)[None, :]
        #     )
        #     env.update_debug_vis()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()