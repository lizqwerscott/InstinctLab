# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export deploy.yaml configuration for a task without training or loading models."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export deploy.yaml configuration for a task.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for deploy.yaml.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.export_deploy_cfg import export_deploy_cfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Export deploy.yaml configuration for a task."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl (just for compatibility)
    env = InstinctRlVecEnvWrapper(env)

    # ensure output directory exists
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # export deploy configuration
    print(f"[INFO] Exporting deploy.yaml configuration for task: {args_cli.task}")
    print(f"[INFO] Output directory: {args_cli.output_dir}")

    export_deploy_cfg(env.unwrapped, args_cli.output_dir)

    print(f"[INFO] Successfully exported deploy.yaml to: {os.path.join(args_cli.output_dir, 'params', 'deploy.yaml')}")

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
