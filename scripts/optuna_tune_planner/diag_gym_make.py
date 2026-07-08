#!/usr/bin/env python3
"""Minimal test: isolate gym.make with increased recursion limit."""

import argparse, os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import instinctlab.tasks  # noqa: F401

import sys as _sys
print(f"[0] Default recursion limit: {_sys.getrecursionlimit()}")
limit = 20000
_sys.setrecursionlimit(limit)
print(f"[0] Set to: {limit}")

from isaaclab_tasks.utils import parse_env_cfg

try:
    print(f"[1] parse_env_cfg (num_envs={args_cli.num_envs}) ...")
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    print("    OK")

    print(f"[2] Current recursion limit before gym.make: {_sys.getrecursionlimit()}")

    print("[3] gym.make ...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("    OK")

    print("[4] InstinctRlVecEnvWrapper ...")
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    env = InstinctRlVecEnvWrapper(env)
    print("    OK")

    print("\n*** ALL OK — gym.make does NOT recurse with this config ***")

finally:
    _sys.setrecursionlimit(1000)
    simulation_app.close()
