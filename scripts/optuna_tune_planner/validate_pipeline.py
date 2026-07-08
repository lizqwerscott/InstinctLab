#!/usr/bin/env python3
"""
validate_pipeline.py — phased validation of the DCM planner tuning pipeline.

This script tests each module in isolation and then tests the integration
points.  It is designed to work on BOTH:

  - A development machine without Isaac Sim (offline checks only).
  - A training server with Isaac Sim running (full integration test).

Usage::

    # Offline validation (no Isaac Sim needed)
    python3 scripts/optuna_tune_planner/validate_pipeline.py

    # Full validation (requires Isaac Sim + a trained checkpoint)
    python3 scripts/optuna_tune_planner/validate_pipeline.py \\
        --checkpoint /path/to/model_5000.pt \\
        --task Instinct-Parkour-Target-Amp-G1-v0 \\
        --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, Dict, List, Tuple

# Ensure the project root is on sys.path so that "from scripts.optuna_tune_planner.xxx"
# imports work regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ==========================================================================
# Utility
# ==========================================================================

PASS = "✓"
FAIL = "✗"
SKIP = "⊘"


def _color(s: str, code: str) -> str:
    """Wrap a string in an ANSI colour code (no-op if stdout is not a tty)."""
    if not sys.stdout.isatty():
        return s
    colours = {"green": "32", "red": "31", "yellow": "33", "bold": "1"}
    c = colours.get(code, "0")
    return f"\033[{c}m{s}\033[0m"


def _print_result(
    step: str, name: str, ok: bool, detail: str = "", skipped: bool = False
) -> None:
    """Print a single test result in a consistent format."""
    if skipped:
        marker = _color(SKIP, "yellow")
    elif ok:
        marker = _color(PASS, "green")
    else:
        marker = _color(FAIL, "red")

    line = f"  [{marker}] {step}: {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


# ==========================================================================
# Phase 0: Offline checks (no Isaac Sim, no Optuna required)
# ==========================================================================

def check_syntax() -> bool:
    """Verify all Python files compile without errors."""
    import py_compile

    print("\n" + "─" * 60)
    print("Phase 0: Offline syntax & import checks")
    print("─" * 60)

    base = os.path.dirname(os.path.abspath(__file__))
    files = [
        "__init__.py", "config.py", "metrics.py",
        "injector.py", "evaluator.py", "tune_planner.py",
        "analyze_results.py",
    ]

    all_ok = True
    for fname in files:
        path = os.path.join(base, fname)
        try:
            py_compile.compile(path, doraise=True)
            _print_result("0.1", fname, True)
        except py_compile.PyCompileError as e:
            _print_result("0.1", fname, False, str(e))
            all_ok = False
    return all_ok


def check_config() -> bool:
    """Validate the search-space and evaluation config."""
    try:
        from scripts.optuna_tune_planner.config import (
            DEFAULT_PARAMS,
            EvalConfig,
            SEARCH_SPACE,
        )
    except ImportError as e:
        _print_result("0.2", "import config", False, str(e))
        return False

    ok = True

    # --- Check search space ---
    expected_params = {
        "alpha_pos", "alpha_dcm", "alpha_E", "alpha_Q",
        "alpha_M", "alpha_climb", "beta", "lp",
    }
    actual_params = set(SEARCH_SPACE.keys())
    if actual_params == expected_params:
        _print_result("0.2", "search space keys", True)
    else:
        missing = expected_params - actual_params
        extra = actual_params - expected_params
        detail = ""
        if missing:
            detail += f"missing: {missing} "
        if extra:
            detail += f"unexpected: {extra}"
        _print_result("0.2", "search space keys", False, detail)
        ok = False

    # --- Check default params match search space ---
    default_keys = set(DEFAULT_PARAMS.keys())
    if default_keys == expected_params:
        _print_result("0.2", "default params keys", True)
    else:
        _print_result("0.2", "default params keys", False,
                      f"mismatch: search={expected_params}, defaults={default_keys}")
        ok = False

    # --- Check each default value is within its search bounds ---
    for name, (dist, lo, hi) in SEARCH_SPACE.items():
        val = DEFAULT_PARAMS.get(name)
        if val is None:
            _print_result("0.2", f"default '{name}' in bounds", False, "missing default")
            ok = False
        elif lo <= val <= hi:
            _print_result("0.2", f"default '{name}'={val} ∈ [{lo}, {hi}]", True)
        else:
            _print_result("0.2", f"default '{name}'={val} ∈ [{lo}, {hi}]", False)
            ok = False

    # --- Check EvalConfig terrain weights sum to ~1 ---
    cfg = EvalConfig()
    w_sum = sum(cfg.terrain_weights.values())
    if abs(w_sum - 1.0) < 0.01:
        _print_result("0.2", f"terrain weights sum={w_sum:.2f}", True)
    else:
        _print_result("0.2", f"terrain weights sum={w_sum:.2f}", False)
        ok = False

    # --- Check composite weights are non-negative ---
    weights = [
        cfg.w_foothold_proximity,
        cfg.w_success_rate,
        cfg.w_tracking_penalty,
        cfg.w_foot_slip_penalty,
    ]
    all_pos = all(w >= 0 for w in weights)
    _print_result("0.2", f"composite weights ≥ 0: {weights}", all_pos)
    if not all_pos:
        ok = False

    return ok


def check_metrics_logic() -> bool:
    """Test MetricsAccumulator with synthetic data."""
    try:
        from scripts.optuna_tune_planner.config import EvalConfig
        from scripts.optuna_tune_planner.metrics import MetricsAccumulator
    except ImportError as e:
        _print_result("0.3", "import metrics", False, str(e))
        return False

    ok = True
    cfg = EvalConfig()
    accum = MetricsAccumulator(cfg)

    # --- Test 1: empty accumulator returns a finite score ---
    score = accum.compute_score()
    if 0.0 <= score <= 1.0:
        _print_result("0.3", "empty score ∈ [0,1]", True, f"score={score:.4f}")
    else:
        _print_result("0.3", "empty score ∈ [0,1]", False, f"score={score:.4f}")
        ok = False

    # --- Test 2: synthetic perfect data returns high score ---
    n_envs = 64
    for _ in range(100):
        accum.update(
            foothold_reward=[0.9] * n_envs,       # high reward
            tracking_error=[0.05] * n_envs,        # low error
            foot_slip=[0.01] * n_envs,             # low slip
            done_mask=[False] * n_envs,            # no terminations
        )
    perfect_score = accum.compute_score()
    if perfect_score > 0.5:
        _print_result("0.3", "perfect data score > 0.5", True, f"score={perfect_score:.4f}")
    else:
        _print_result("0.3", "perfect data score > 0.5", False, f"score={perfect_score:.4f}")
        ok = False

    # --- Test 3: synthetic bad data returns low score ---
    accum2 = MetricsAccumulator(cfg)
    for _ in range(100):
        accum2.update(
            foothold_reward=[0.1] * n_envs,       # low reward
            tracking_error=[1.5] * n_envs,         # high error
            foot_slip=[0.1] * n_envs,              # high slip
            done_mask=[True] * (n_envs // 2) + [False] * (n_envs // 2),  # 50% deaths
        )
    bad_score = accum2.compute_score()
    if bad_score < perfect_score:
        _print_result("0.3", "bad < perfect score", True,
                      f"bad={bad_score:.4f} < perfect={perfect_score:.4f}")
    else:
        _print_result("0.3", "bad < perfect score", False,
                      f"bad={bad_score:.4f} ≥ perfect={perfect_score:.4f}")
        ok = False

    # --- Test 4: summary() returns expected keys ---
    summ = accum.summary()
    expected_summary_keys = {
        "foothold_reward_mean", "success_rate",
        "tracking_error_mean", "composite_score",
    }
    has_keys = expected_summary_keys.issubset(set(summ.keys()))
    _print_result("0.3", "summary() has expected keys", has_keys,
                  f"keys={list(summ.keys())}")
    if not has_keys:
        ok = False

    return ok


# ==========================================================================
# Phase 1: Isaac Sim checks (requires AppLauncher to be running)
# ==========================================================================

def check_isaac_imports() -> Tuple[bool, bool]:
    """Check whether Isaac Lab and instinctlab can be imported.

    Returns:
        (ok, available): ``ok`` is always True for this check (it just
        reports status); ``available`` is True if the packages are importable.
    """
    print("\n" + "─" * 60)
    print("Phase 1: Isaac Lab / InstinctLab availability")
    print("─" * 60)

    available = True
    for mod_name in ["isaaclab", "instinctlab", "optuna"]:
        try:
            __import__(mod_name)
            _print_result("1.0", f"import {mod_name}", True)
        except ImportError:
            _print_result("1.0", f"import {mod_name}", False, "not installed in this environment")
            available = False
    return available, available  # first bool is "test passed" (always true), second is "available"


def check_env_creation(task_name: str) -> bool:
    """Test that the gym environment can be created and wrapped."""
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
    except ImportError as e:
        _print_result("1.1", "env creation imports", False, str(e))
        return False

    ok = True
    try:
        env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)
        _print_result("1.1", "parse_env_cfg", True)
    except Exception as e:
        _print_result("1.1", "parse_env_cfg", False, str(e)[:100])
        return False

    try:
        env_cfg.scene.num_envs = 4
        # Disable camera to speed up test.
        if hasattr(env_cfg.scene, "camera"):
            env_cfg.scene.camera = None
        env = gym.make(task_name, cfg=env_cfg)
        _print_result("1.1", "gym.make", True, f"num_envs={env.unwrapped.num_envs}")
    except Exception as e:
        _print_result("1.1", "gym.make", False, str(e)[:150])
        return False

    try:
        env = InstinctRlVecEnvWrapper(env)
        obs, _ = env.get_observations()
        _print_result("1.1", "InstinctRlVecEnvWrapper", True,
                      f"obs keys={list(obs.keys())}")
    except Exception as e:
        _print_result("1.1", "InstinctRlVecEnvWrapper", False, str(e)[:150])
        ok = False

    try:
        env.close()
    except Exception:
        pass
    return ok


def check_reward_term_exists(task_name: str) -> bool:
    """Verify the foothold_proximity reward term exists in the env."""
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
    except ImportError:
        return False

    try:
        env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)
        env_cfg.scene.num_envs = 4
        if hasattr(env_cfg.scene, "camera"):
            env_cfg.scene.camera = None
        env = gym.make(task_name, cfg=env_cfg)

        reward_mgr = env.unwrapped.reward_manager
        term_names = getattr(reward_mgr, "_term_names", [])
        has_term = "foothold_proximity" in term_names
        _print_result("1.2", "foothold_proximity term exists", has_term,
                      f"available terms: {term_names}")
        env.close()
        return has_term
    except Exception as e:
        _print_result("1.2", "foothold_proximity term check", False, str(e)[:150])
        return False


# ==========================================================================
# Phase 2: PlannerInjector integration test
# ==========================================================================

def check_injector(task_name: str) -> bool:
    """Test that PlannerInjector can swap the planner and restore it."""
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
        from scripts.optuna_tune_planner.config import DEFAULT_PARAMS
        from scripts.optuna_tune_planner.injector import PlannerInjector
        from instinctlab.tasks.parkour.mdp.dcm_planner import DCMFootholdPlanner
    except ImportError as e:
        _print_result("2.0", "injector imports", False, str(e))
        return False

    ok = True
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)
    env_cfg.scene.num_envs = 4
    if hasattr(env_cfg.scene, "camera"):
        env_cfg.scene.camera = None
    env = gym.make(task_name, cfg=env_cfg)
    env = InstinctRlVecEnvWrapper(env)

    try:
        # --- Test 1: enter swaps the planner ---
        reward_mgr = env.unwrapped.reward_manager
        term_idx = reward_mgr._term_names.index("foothold_proximity")
        original_planner = reward_mgr._terms[term_idx]._planner

        # Modify one parameter to a value we can verify.
        test_params = DEFAULT_PARAMS.copy()
        test_params["alpha_pos"] = 9.99  # deliberately obvious

        with PlannerInjector(env, test_params) as inj:
            new_planner = reward_mgr._terms[term_idx]._planner
            if new_planner is not original_planner:
                _print_result("2.1", "planner swapped on enter", True)
            else:
                _print_result("2.1", "planner swapped on enter", False, "same object")
                ok = False

            if abs(new_planner.alpha_pos - 9.99) < 0.001:
                _print_result("2.1", f"custom alpha_pos={new_planner.alpha_pos}", True)
            else:
                _print_result("2.1", "custom alpha_pos applied", False,
                              f"expected 9.99, got {new_planner.alpha_pos}")
                ok = False

        # --- Test 2: exit restores original ---
        restored_planner = reward_mgr._terms[term_idx]._planner
        if restored_planner is original_planner:
            _print_result("2.2", "planner restored on exit", True)
        else:
            _print_result("2.2", "planner restored on exit", False, "different object")
            ok = False

        # --- Test 3: exception safety (exit still runs) ---
        cleanup_ran = False
        try:
            with PlannerInjector(env, test_params):
                cleanup_ran = True
                raise RuntimeError("simulated error inside injector block")
        except RuntimeError:
            pass
        final_planner = reward_mgr._terms[term_idx]._planner
        if final_planner is original_planner and cleanup_ran:
            _print_result("2.3", "exception-safe restore", True)
        else:
            _print_result("2.3", "exception-safe restore", False)
            ok = False

    finally:
        env.close()
    return ok


# ==========================================================================
# Phase 3: Full rollout test (requires checkpoint)
# ==========================================================================

def check_rollout(task_name: str, checkpoint_path: str, num_envs: int = 16) -> bool:
    """Run a short rollout with the baseline planner and verify metrics come out."""
    try:
        import numpy as np
        from scripts.optuna_tune_planner.config import DEFAULT_PARAMS, EvalConfig
        from scripts.optuna_tune_planner.evaluator import PlannerEvaluator
    except ImportError as e:
        _print_result("3.0", "evaluator imports", False, str(e))
        return False

    ok = True
    cfg = EvalConfig()
    cfg.num_envs = num_envs
    cfg.rollout_steps = 50  # short for quick validation
    cfg.num_repeat = 1

    # 3.1 Create evaluator (loads env + policy)
    try:
        evaluator = PlannerEvaluator(cfg, task_name, checkpoint_path)
        _print_result("3.1", "PlannerEvaluator created", True)
    except Exception as e:
        _print_result("3.1", "PlannerEvaluator created", False, str(e)[:200])
        return False

    # 3.2 Baseline rollout
    try:
        score, std = evaluator.evaluate_baseline(num_repeat=2)
        _print_result("3.2", "baseline rollout", True,
                      f"score={score:.4f} ± {std:.4f}")
        if score < 0.0 or score > 1.0:
            _print_result("3.2", "baseline score ∈ [0,1]", False, f"score={score:.4f}")
            ok = False
    except Exception as e:
        _print_result("3.2", "baseline rollout", False, traceback.format_exc()[:300])
        ok = False

    # 3.3 Rollout with modified params (should produce a different score)
    try:
        modified_params = DEFAULT_PARAMS.copy()
        modified_params["alpha_Q"] = 50.0  # extreme: punish roughness heavily
        modified_params["alpha_pos"] = 5.0  # extreme: force nominal position
        mod_score, mod_std = evaluator.evaluate(modified_params)
        _print_result("3.3", "modified-params rollout", True,
                      f"score={mod_score:.4f} ± {mod_std:.4f}")

        delta = abs(mod_score - score)
        if delta > 0.001:
            _print_result("3.3", "score differs from baseline", True, f"Δ={delta:.4f}")
        else:
            _print_result("3.3", "score differs from baseline", False,
                          f"Δ={delta:.4f} — planner params may not affect score")
            ok = False
    except Exception as e:
        _print_result("3.3", "modified-params rollout", False, traceback.format_exc()[:300])
        ok = False

    evaluator.close()
    return ok


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the DCM planner tuning pipeline."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a trained .pt checkpoint (required for Phase 3).",
    )
    parser.add_argument(
        "--task", type=str, default="Instinct-Parkour-Target-Amp-G1-v0",
        help="Gym task id.",
    )
    parser.add_argument(
        "--online", action="store_true", default=False,
        help="Run Phase 1-3 checks (requires Isaac Sim).",
    )
    # ----------------------------------------------------------------
    # Isaac Sim AppLauncher args — only added when isaaclab is installed.
    # ----------------------------------------------------------------
    try:
        from isaaclab.app import AppLauncher as _AppLauncher
        _AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        pass  # isaaclab not installed — AppLauncher args are not available
    args_cli, _ = parser.parse_known_args()

    # ==================================================================
    # Header
    # ==================================================================
    print("=" * 60)
    print("  DCM PLANNER TUNING — PIPELINE VALIDATION")
    print("=" * 60)
    print(f"  Python:  {sys.executable}")
    print(f"  CWD:     {os.getcwd()}")
    print(f"  Task:    {args_cli.task}")
    print(f"  Checkpoint: {args_cli.checkpoint or '(not provided)'}")

    results: List[Tuple[str, bool, bool]] = []  # (name, ok, skipped)

    # ==================================================================
    # Phase 0: Offline (always runs)
    # ==================================================================
    ok = check_syntax()
    results.append(("syntax", ok, False))

    ok = check_config()
    results.append(("config", ok, False))

    ok = check_metrics_logic()
    results.append(("metrics", ok, False))

    # ==================================================================
    # Phase 1-3: Online checks (require Isaac Sim running)
    # ==================================================================
    if not args_cli.online:
        print(f"\n  {_color(SKIP, 'yellow')} Phase 1-3 skipped (add --online to run Isaac Sim checks)")
        results.append(("isaac-imports", False, True))
        results.append(("env-creation", False, True))
        results.append(("reward-term", False, True))
        results.append(("injector", False, True))
        results.append(("rollout", False, True))
        _print_summary(results, 0, 5)
        return

    # ---- Start Isaac Sim first (like train.py does) ----
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Now it is safe to import pxr / instinctlab / gym.
    # Import instinctlab.tasks to register gym environments.
    import gymnasium as gym
    import instinctlab.tasks  # noqa: F401 — side-effect: registers envs

    try:
        ok = check_isaac_imports()[0]
        results.append(("isaac-imports", ok, False))

        ok = check_env_creation(args_cli.task)
        results.append(("env-creation", ok, False))

        ok = check_reward_term_exists(args_cli.task)
        results.append(("reward-term", ok, False))

        ok = check_injector(args_cli.task)
        results.append(("injector", ok, False))

        if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
            ok = check_rollout(args_cli.task, args_cli.checkpoint)
            results.append(("rollout", ok, False))
        elif args_cli.checkpoint:
            print(f"\n  {_color(SKIP, 'yellow')} Checkpoint not found: {args_cli.checkpoint}")
            results.append(("rollout", False, True))
        else:
            print(f"\n  {_color(SKIP, 'yellow')} Phase 3 skipped (no --checkpoint provided)")
            results.append(("rollout", False, True))
    finally:
        # Always close the simulation app.
        simulation_app.close()

    _print_summary(results, 0, 0)


def _print_summary(
    results: List[Tuple[str, bool, bool]],
    _unused_pass: int,
    _unused_skip: int,
) -> None:
    """Print the final summary table and exit with appropriate code."""
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for _, ok, skipped in results if ok and not skipped)
    n_fail = sum(1 for _, ok, skipped in results if not ok and not skipped)
    n_skip = sum(1 for _, _, skipped in results if skipped)

    for name, ok, skipped in results:
        if skipped:
            marker = _color(SKIP + " SKIP", "yellow")
        elif ok:
            marker = _color(PASS + " PASS", "green")
        else:
            marker = _color(FAIL + " FAIL", "red")
        print(f"  {marker}  {name}")

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")

    if n_fail > 0:
        print(f"\n  {_color('Some checks FAILED. Review the output above.', 'red')}")
        sys.exit(1)
    elif n_skip > 0 and n_pass > 0:
        print(f"\n  {_color('Offline checks passed. Run with --online --checkpoint <path> for full validation.', 'yellow')}")
    else:
        print(f"\n  {_color('All checks passed!', 'green')}")
        print(f"  Ready to run: python scripts/optuna_tune_planner/tune_planner.py --checkpoint ...")


if __name__ == "__main__":
    main()
