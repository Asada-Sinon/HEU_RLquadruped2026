# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

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
parser.add_argument("--trace", type=str, default=None, help="Optional .npz path for saving policy-step trace data.")
parser.add_argument("--trace_length", type=int, default=500, help="Number of policy steps to save when --trace is set.")
parser.add_argument(
    "--trace_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Force a fixed base_velocity command while tracing.",
)
parser.add_argument(
    "--trace_ground_box",
    action="store_true",
    default=False,
    help="Use a simple collision cuboid as ground while tracing, bypassing the default ground-plane USD spawner.",
)
parser.add_argument(
    "--trace_nominal",
    action="store_true",
    default=False,
    help="Disable observation corruption and randomization while tracing for a cleaner sim2sim comparison.",
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
import numpy as np
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_first.tasks  # noqa: F401


class TraceBuffer:
    """Small NPZ trace writer for one-env policy debugging."""

    def __init__(self, metadata: dict):
        self.metadata = metadata
        self._data = {}

    def append(self, values: dict):
        for key, value in values.items():
            self._data.setdefault(key, []).append(np.asarray(value).copy())

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {key: np.stack(values, axis=0) for key, values in self._data.items() if values}
        for key, value in self.metadata.items():
            arrays[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(path, **arrays)
        print(f"[INFO] Wrote trace: {os.path.abspath(path)}")


def _to_np_first(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy()


def _force_base_velocity_command(env, command: torch.Tensor | None):
    if command is None or not hasattr(env.unwrapped, "command_manager"):
        return
    command_manager = env.unwrapped.command_manager
    if "base_velocity" not in command_manager.active_terms:
        return
    term = command_manager.get_term("base_velocity")
    term.vel_command_b[:, :] = command.to(device=term.vel_command_b.device, dtype=term.vel_command_b.dtype)
    term.is_heading_env[:] = False
    term.is_standing_env[:] = False
    term.time_left[:] = 1.0e9


def _collect_trace_step(env, obs, actions, timestep: int) -> dict:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    robot_data = robot.data
    action_term = base_env.action_manager.get_term("joint_pos")
    command = base_env.command_manager.get_command("base_velocity")
    return {
        "time": np.array(timestep * base_env.step_dt, dtype=np.float64),
        "command": _to_np_first(command),
        "obs": _to_np_first(obs["policy"]),
        "raw_action": _to_np_first(actions),
        "target_joint_pos": _to_np_first(action_term.processed_actions),
        "applied_torque": _to_np_first(robot_data.applied_torque),
        "computed_torque": _to_np_first(robot_data.computed_torque),
        "base_pos": _to_np_first(robot_data.root_pos_w),
        "base_quat": _to_np_first(robot_data.root_quat_w),
        "base_lin_vel": _to_np_first(robot_data.root_lin_vel_b),
        "base_ang_vel": _to_np_first(robot_data.root_ang_vel_b),
        "projected_gravity": _to_np_first(robot_data.projected_gravity_b),
        "joint_pos": _to_np_first(robot_data.joint_pos),
        "joint_pos_rel": _to_np_first(robot_data.joint_pos - robot_data.default_joint_pos),
        "joint_vel": _to_np_first(robot_data.joint_vel),
    }


def _replace_terrain_with_ground_box(env_cfg):
    env_cfg.scene.terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(100.0, 100.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        collision_group=-1,
    )
    if getattr(env_cfg.scene, "height_scanner", None) is not None:
        env_cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground/geometry/mesh"]


def _disable_trace_only_visualizers(env_cfg):
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.debug_vis = False


def _make_trace_nominal(env_cfg):
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg, "events"):
        for name in ("physics_material", "add_base_mass", "base_com", "push_robot"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
            env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
            env_cfg.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
            env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.trace:
        env_cfg.scene.num_envs = 1
        _disable_trace_only_visualizers(env_cfg)
    if args_cli.trace_nominal:
        _make_trace_nominal(env_cfg)
    if args_cli.trace_ground_box:
        _replace_terrain_with_ground_box(env_cfg)

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
            "video_folder": os.path.join(log_dir, "videos", "play"),
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
    fixed_trace_command = None
    if args_cli.trace_command is not None:
        fixed_trace_command = torch.tensor(args_cli.trace_command, device=env.unwrapped.device).repeat(env.num_envs, 1)

    trace = None
    if args_cli.trace:
        robot = env.unwrapped.scene["robot"]
        trace = TraceBuffer(
            metadata={
                "source": "isaaclab",
                "checkpoint": resume_path,
                "joint_names": robot.data.joint_names,
                "default_joint_pos": _to_np_first(robot.data.default_joint_pos),
                "step_dt": dt,
                "physics_dt": env.unwrapped.physics_dt,
            }
        )

    # reset environment
    _force_base_velocity_command(env, fixed_trace_command)
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            obs_for_policy = obs
            # agent stepping
            actions = policy(obs_for_policy)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            _force_base_velocity_command(env, fixed_trace_command)
            if trace is not None:
                trace.append(_collect_trace_step(env, obs_for_policy, actions, timestep))
                obs = env.get_observations()
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if trace is not None and timestep >= args_cli.trace_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if trace is not None:
        trace.save(args_cli.trace)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
