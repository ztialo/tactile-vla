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
        
    else:
        print("invalid control_type: ", control_type)
        raise ValueError(f"Invalid control type: {control_type}")

    cfg = SingleArmEnvCfg()
    env = SingleArmEnv(cfg)

    if obstacle:
        env.set_obstacle()

    env.reset()
    count = 0

    output_dir = os.path.join(root_dir, "output_infer")
    os.makedirs(output_dir, exist_ok=True)
    episode_id = 0
    episode_data = EpisodeData(episode_id=episode_id, episode_data={})

    is_recording = False
    action_queue = None
    while simulation_app.is_running() and episode_id < num_max_episodes:
        if control_type == "gello":
            gello_qpos = gello.get_gello_action()
            gello_qpos = gello_qpos[:7]
            action_queue = gello_qpos[None, :]
        elif control_type == "ik":
            task_state = env.get_task_state()
            state_machine.update_sm(task_state)
            goal_pose = state_machine.get_goal_pose(task_state)
            gello_qpos = env.compute_ik(goal_pose).squeeze(0).cpu().numpy()
            action_queue = gello_qpos[None, :]
        elif control_type == "pyroki":
            task_state = env.get_task_state()
            state_machine.update_sm(task_state)
            goal_pose = state_machine.get_goal_pose(task_state)
            action_queue = env.solve_trajectory(goal_pose, state_machine.state)
        elif control_type == "infer":
            while True:
                try:
                    ws = websocket.create_connection(SERVER_URL)
                    break
                except Exception as e:
                    print(f"Trying to connect to server: {e}")
                    time.sleep(1)
            print("Connected to server!")
            obs = env._get_observations()
            state = obs["/action/qpos"]
            images = {
                "top": obs["/observations/camera_top"],
                "hand": obs["/observations/camera_hand"],
            }
            prompt = "pick up the object"
            ws.send(
                create_payload(images, state, prompt),
                opcode=websocket.ABNF.OPCODE_BINARY,
            )
            response = ws.recv()
            action_list = json.loads(response)["actions"]
            gello_qpos = np.stack([step["action"] for step in action_list])
            num_run = 20  # Run the first 20 actions
            action_queue = gello_qpos[:num_run]
            ws.close()
        else:
            raise ValueError(f"Invalid control type: {control_type}")

        if control_type == "gello":
            if np.all(np.abs(action_queue[0, :6]) < 0.15):
                # Arm is set to the initial position
                env.set_unlock()
                is_recording = True
            elif env.is_locked.any():
                if count % 5 == 0:
                    print("------------- Aligning Gello -------------------")
                    for j_idx, j_value in enumerate(action_queue[0, :6]):
                        print(f"Joint {j_idx}: {j_value:.3f}")
        else:
            env.set_unlock()
            is_recording = True

        # Loop over action queue
        for gello_qpos in action_queue:
            obs, rew, done, truncated, extras = env.step(
                torch.from_numpy(gello_qpos[:7]).to(env.device)[None, :]
            )
            env.update_debug_vis()

        # temporary debug block
        # if control_type == "pyroki":
        # env.print_debug_messages(goal_pose)

        count += 1
        if done.any() and not episode_data.is_empty():
            output_file = os.path.join(output_dir, f"episode_data_{episode_id:06d}.h5")
            episode_data.to_h5(output_file)
            print(f"[INFO] Saved episode data to {output_file}")
            episode_id += 1  # Increment the episode id
        if done.any() or truncated.any():
            # Reset the episode data
            episode_data.clear()
            episode_data.reset(episode_id)
            # Reset the state machine
            if control_type == "ik" or control_type == "pyroki":
                state_machine.reset()
            count = 0
            is_recording = False
        else:
            # Collect data
            if is_recording:
                episode_data.add_frame_data(obs)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()