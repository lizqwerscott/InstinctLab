#!/usr/bin/env python3
"""Deep diagnostic: inspect MultiRewardManager group structure."""

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
_sys.setrecursionlimit(5000)

from isaaclab_tasks.utils import parse_env_cfg

try:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=4)
    env_cfg.scene.num_envs = 4
    env = gym.make(args_cli.task, cfg=env_cfg)
    mgr = env.unwrapped.reward_manager

    # ---- 1. Look inside __group_term_cfgs ----
    group_cfgs = getattr(mgr, "_MultiRewardManager__group_term_cfgs", {})
    print("__group_term_cfgs keys:", list(group_cfgs.keys()))
    for gname, gcfg in group_cfgs.items():
        print(f"  [{gname}]")
        if hasattr(gcfg, '__dataclass_fields__'):
            for fname in gcfg.__dataclass_fields__:
                val = getattr(gcfg, fname, None)
                if hasattr(val, '__dataclass_fields__'):
                    terms = list(val.__dataclass_fields__.keys())
                    print(f"    {fname}: dataclass with fields: {terms}")
                    if "foothold_proximity" in terms:
                        print(f"      *** FOUND foothold_proximity in {gname}.{fname} ***")
                else:
                    print(f"    {fname}: {type(val).__name__}")

    # ---- 2. Look inside __group_term_names ----
    group_names = getattr(mgr, "_MultiRewardManager__group_term_names", {})
    print("\n__group_term_names:", group_names)

    # ---- 3. Look for any attribute containing term instances ----
    for attr in dir(mgr):
        if attr.startswith('_'):
            try:
                val = getattr(mgr, attr)
                if isinstance(val, dict):
                    for k, v in val.items():
                        if hasattr(v, '_planner'):
                            print(f"\n  *** FOUND _planner in: mgr.{attr}['{k}']")
                            print(f"      type={type(v).__name__}")
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                if hasattr(v2, '_planner'):
                                    print(f"\n  *** FOUND _planner in: mgr.{attr}['{k}']['{k2}']")
                                    print(f"      type={type(v2).__name__}")
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        if hasattr(item, '_planner'):
                            print(f"\n  *** FOUND _planner in: mgr.{attr}[{i}]")
                            print(f"      type={type(item).__name__}")
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if hasattr(v, '_planner'):
                                    print(f"\n  *** FOUND _planner in: mgr.{attr}[{i}]['{k}']")
                                    print(f"      type={type(v).__name__}")
            except Exception:
                pass

    # ---- 4. Dump _episode_sums keys to understand naming convention ----
    ep_sums = getattr(mgr, "_episode_sums", {})
    foothold_keys = [k for k in ep_sums if 'foothold' in k.lower()]
    print(f"\n_episode_sums keys containing 'foothold': {foothold_keys}")

    # ---- 5. Try the compute() path ----
    print("\n[5] Checking if term exists via cfg...")
    cfg = getattr(mgr, "cfg", None)
    print(f"    cfg type: {type(cfg).__name__}")
    if hasattr(cfg, 'rewards') and hasattr(cfg.rewards, '__dataclass_fields__'):
        fields = list(cfg.rewards.__dataclass_fields__.keys())
        print(f"    cfg.rewards fields: {fields}")
        print(f"    foothold_proximity in cfg.rewards: {'foothold_proximity' in fields}")

finally:
    _sys.setrecursionlimit(_old)
    simulation_app.close()
