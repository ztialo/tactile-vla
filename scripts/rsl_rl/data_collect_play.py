# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args
# from source.fr3_manipulation.fr3_manipulation.tasks.manager_based.fr3_manipulation import env, robot  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log", action="store_true", default=False, help="Log training information.")
parser.add_argument(
    "--action_smoothing_alpha",
    type=float,
    default=0.3,
    help="Low-pass smoothing factor for actions in [0, 1]. 1.0 disables smoothing.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_manipulation.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    ft_log_dir = os.path.join("logs", "ft_sensor")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(ft_log_dir, "videos", datetime.now().strftime("%Y%m%d_%H%M%S")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # FT sensor logging setup (IsaacLab API: use scene articulation handle directly)
    robot = env.unwrapped.scene["robot"]
    left_ft_body_name = "fr3_left_ft"
    right_ft_body_name = "fr3_right_ft"
    left_ft_body_idx = None
    right_ft_body_idx = None
    try:
        left_ids, _ = robot.find_bodies(left_ft_body_name)
        right_ids, _ = robot.find_bodies(right_ft_body_name)
        if len(left_ids) > 0 and len(right_ids) > 0:
            left_ft_body_idx = left_ids[0]
            right_ft_body_idx = right_ids[0]
        else:
            print(
                f"[WARN] FT bodies not found on robot. Expected '{left_ft_body_name}' and "
                f"'{right_ft_body_name}'. FT logging will write empty values."
            )
    except Exception as exc:
        print(f"[WARN] Failed to resolve FT body indices: {exc}. FT logging will write empty values.")

    def _serialize_ft_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    # reset environment
    obs = env.get_observations()
    timestep = 0
    count = 0

    if not (0.0 <= args_cli.action_smoothing_alpha <= 1.0):
        raise ValueError("--action_smoothing_alpha must be in [0, 1].")
    action_smoothing_alpha = args_cli.action_smoothing_alpha
    last_actions = None
    
    ft_csv_file = None
    ft_writer = None
    if args_cli.log:
        os.makedirs(ft_log_dir, exist_ok=True)
        ft_log_path = os.path.join(ft_log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        ft_csv_file = open(ft_log_path, "w", newline="")
        ft_writer = csv.writer(ft_csv_file)
        ft_writer.writerow(["wall_time_iso", "step", "left_ft_joint", "right_ft_joint"])
        print(f"[INFO] FT log file: {ft_log_path}")

    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                raw_actions = policy(obs)
                if last_actions is None or action_smoothing_alpha >= 1.0:
                    actions = raw_actions
                else:
                    # First-order low-pass filter: smooth high-frequency action changes.
                    actions = action_smoothing_alpha * raw_actions + (1.0 - action_smoothing_alpha) * last_actions
                    # Reset filter state for done envs to avoid carrying stale actions across episode resets.
                    if dones is not None and torch.any(dones):
                        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                        actions[done_ids] = raw_actions[done_ids]
                last_actions = actions
                # env stepping
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            if count % 1 == 0:  # publish at every timestep so is 60
                if args_cli.log and ft_writer is not None:
                    if left_ft_body_idx is not None and right_ft_body_idx is not None:
                        # Shape: (num_envs, num_bodies, 6). We log env-0 wrench [Fx, Fy, Fz, Tx, Ty, Tz].
                        body_wrenches = robot.data.body_incoming_joint_wrench_b
                        left_val = body_wrenches[0, left_ft_body_idx]
                        right_val = body_wrenches[0, right_ft_body_idx]
                    else:
                        left_val = None
                        right_val = None
                    ft_writer.writerow(
                        [
                            datetime.now().isoformat(timespec="milliseconds"),
                            count,
                            _serialize_ft_value(left_val),
                            _serialize_ft_value(right_val),
                        ]
                    )
                    ft_csv_file.flush()

            # update counter for FT sensor logging
            count += 1

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if ft_csv_file is not None:
            ft_csv_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
