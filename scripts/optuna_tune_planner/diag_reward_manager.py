#!/usr/bin/env python3
"""Quick diagnostic: dump the reward manager's internal attributes.

Run with Isaac Sim::

    python scripts/optuna_tune_planner/diag_reward_manager.py \\
        --task Instinct-Parkour-Target-Amp-G1-v0 --headless
"""

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
import instinctlab.tasks  # noqa: F401 — register envs

import sys as _sys
_old = _sys.getrecursionlimit()
_sys.setrecursionlimit(5000)

from isaaclab_tasks.utils import parse_env_cfg

try:
    print("[1] parse_env_cfg ...")
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=4)
    env_cfg.scene.num_envs = 4
    print("    OK")

    print("[2] gym.make ...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("    OK")

    print("[3] Inspecting reward_manager ...")
    mgr = env.unwrapped.reward_manager
    print(f"    type(mgr) = {type(mgr).__name__}")

    # Dump ALL attributes (both public and private)
    all_attrs = sorted([a for a in dir(mgr) if not a.startswith('__')])
    print(f"    all attrs ({len(all_attrs)}):")
    for attr in all_attrs:
        try:
            val = getattr(mgr, attr)
            if isinstance(val, dict):
                print(f"      {attr}: dict with keys={list(val.keys())[:10]}")
            elif isinstance(val, list):
                print(f"      {attr}: list len={len(val)}")
            elif callable(val):
                pass  # skip methods
            else:
                rep = repr(val)
                if len(rep) > 80:
                    rep = rep[:80] + "..."
                print(f"      {attr}: {rep}")
        except Exception as e:
            print(f"      {attr}: ERROR={e}")

    # Specifically check for foothold_proximity
    print()
    print("[4] Searching for 'foothold_proximity' ...")
    found = False
    for attr in all_attrs:
        try:
            val = getattr(mgr, attr)
            if isinstance(val, dict) and "foothold_proximity" in val:
                print(f"    FOUND in mgr.{attr} keys")
                found = True
            elif isinstance(val, list) and "foothold_proximity" in val:
                print(f"    FOUND in mgr.{attr} list")
                found = True
            elif isinstance(val, str) and "foothold_proximity" in val:
                print(f"    mgr.{attr} = {val}")
        except Exception:
            pass

    if not found:
        print("    NOT FOUND in any attribute!")
        # Try iterating through terms list
        terms = getattr(mgr, "_terms", [])
        print(f"    _terms has {len(terms)} entries")
        for i, t in enumerate(terms):
            print(f"      [{i}] type={type(t).__name__}, has _planner={hasattr(t, '_planner')}")

finally:
    _sys.setrecursionlimit(_old)
    simulation_app.close()
