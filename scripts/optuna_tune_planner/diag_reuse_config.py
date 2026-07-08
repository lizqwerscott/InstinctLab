#!/usr/bin/env python3
"""Test: does reusing a config across two gym.make calls trigger recursion?"""

import argparse, os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import instinctlab.tasks  # noqa: F401

import sys as _sys
_old = _sys.getrecursionlimit()
_sys.setrecursionlimit(10000)

from isaaclab_tasks.utils import parse_env_cfg

try:
    # --- Pass 1: create with 4 envs, then close ---
    print("[1] parse_env_cfg + gym.make (num_envs=4) ...")
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=4)
    cfg.scene.num_envs = 4
    env1 = gym.make(args_cli.task, cfg=cfg)
    print("    Pass 1 OK")
    env1.close()
    print("    Closed")

    # --- Pass 2: reuse SAME config, different num_envs ---
    print("[2] Reusing same config object, gym.make (num_envs=16) ...")
    cfg.scene.num_envs = 16
    env2 = gym.make(args_cli.task, cfg=cfg)
    print("    Pass 2 OK")
    env2.close()

    # --- Pass 3: fresh config ---
    print("[3] Fresh config, gym.make (num_envs=16) ...")
    cfg2 = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=16)
    cfg2.scene.num_envs = 16
    env3 = gym.make(args_cli.task, cfg=cfg2)
    print("    Pass 3 OK")
    env3.close()

    print("\n*** ALL PASSED ***")

except Exception as e:
    print(f"    FAILED: {e}")

finally:
    _sys.setrecursionlimit(_old)
    simulation_app.close()
