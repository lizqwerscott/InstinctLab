#!/usr/bin/env python3
"""
tune_planner.py — Bayesian optimisation of DCMFootholdPlanner cost weights.

Run with::

    python scripts/optuna_tune_planner/tune_planner.py \\
        --task Instinct-Parkour-Target-Amp-G1-v0 \\
        --checkpoint /path/to/model.pt \\
        --n_trials 100

The script:
  1. Launches Isaac Sim via ``AppLauncher``.
  2. Creates a lightweight evaluation environment and loads a frozen policy.
  3. Runs an Optuna study (TPE sampler + MedianPruner) over the planner's
     cost-channel weights.
  4. Prints the best parameters and optionally persists the study database.

Architecture note:  this script is the "main" entry point.  It depends on
Isaac Sim being available (imported after ``AppLauncher`` starts the
simulation app).  All heavy lifting is delegated to ``PlannerEvaluator``.
"""

from __future__ import annotations

import argparse
import os
import os
import sys
import time
from typing import Any, Dict, Tuple

# Ensure the project root is on sys.path so that "from scripts.optuna_tune_planner.xxx"
# imports work regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ==========================================================================
# Phase 0: parse CLI args *before* importing Isaac Sim
# ==========================================================================
parser = argparse.ArgumentParser(
    description="Bayesian optimisation of DCM foothold planner weights."
)
parser.add_argument(
    "--task",
    type=str,
    default="Instinct-Parkour-Target-Amp-G1-v0",
    help="Gym task id (must be a parkour variant with foothold_proximity reward).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a .pt checkpoint from instinct_rl training.",
)
parser.add_argument(
    "--n_trials",
    type=int,
    default=100,
    help="Total number of Optuna trials (including warmup).",
)
parser.add_argument(
    "--n_startup_trials",
    type=int,
    default=15,
    help="Number of initial random-sampler trials before TPE takes over.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=512,
    help="Number of parallel environments for evaluation rollouts.",
)
parser.add_argument(
    "--rollout_steps",
    type=int,
    default=500,
    help="Simulation steps per rollout.",
)
parser.add_argument(
    "--num_repeat",
    type=int,
    default=3,
    help="Independent rollouts per parameter set (averaged to reduce noise).",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=8 * 3600,
    help="Maximum wall-clock time for the study (seconds).",
)
parser.add_argument(
    "--storage",
    type=str,
    default=None,
    help="Optuna storage URL.  E.g. 'sqlite:///tune_planner.db' to persist across runs.",
)
parser.add_argument(
    "--study_name",
    type=str,
    default="dcm_planner_tune",
    help="Optuna study name (for dashboard / resume).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for the TPE sampler.",
)
# Isaac Sim AppLauncher args (headless, device, etc.)
# NOTE: do NOT add --device yourself — AppLauncher owns it.
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()

# ==========================================================================
# Phase 1: launch Isaac Sim
# ==========================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can safely import the rest.
import numpy as np
import torch

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Register the parkour task (side-effect: adds gym environments).
import instinctlab.tasks  # noqa: F401

from scripts.optuna_tune_planner.config import (
    DEFAULT_PARAMS,
    EvalConfig,
    OptunaConfig,
    SEARCH_SPACE,
)
from scripts.optuna_tune_planner.evaluator import PlannerEvaluator
from scripts.optuna_tune_planner.metrics import MetricsAccumulator


# ==========================================================================
# Phase 2: build the Optuna objective
# ==========================================================================

class _Objective:
    """Callable that wraps ``PlannerEvaluator.evaluate()`` for Optuna.

    Splitting this into a class (instead of a closure) makes it easier to
    pass around the evaluator reference and to add per-trial logging.
    """

    def __init__(
        self,
        evaluator: PlannerEvaluator,
        eval_cfg: EvalConfig,
    ) -> None:
        self._evaluator = evaluator
        self._eval_cfg = eval_cfg
        # Keep a running history for the MedianPruner baseline.
        self._trial_scores: list[float] = []
        self._trial_count: int = 0

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample parameters, evaluate, and return the composite score.

        Called by ``study.optimize()`` for each trial.
        """
        # ---- 1. Sample parameters from the search space ----
        params = _sample_params(trial)

        # ---- 2. Log sampled values ----
        for key, value in params.items():
            trial.set_user_attr(f"param_{key}", value)

        # ---- 3. Evaluate ----
        t_start = time.perf_counter()
        mean_score, std_score = self._evaluator.evaluate(params)
        elapsed = time.perf_counter() - t_start

        # ---- 4. Report to Optuna ----
        trial.set_user_attr("score_std", std_score)
        trial.set_user_attr("elapsed_sec", elapsed)

        self._trial_count += 1
        self._trial_scores.append(mean_score)

        # ---- 5. Per-trial console output ----
        best_so_far = max(self._trial_scores) if self._trial_scores else mean_score
        is_best = mean_score >= best_so_far * 0.999  # allow tiny epsilon

        marker = " *** NEW BEST ***" if is_best else ""
        print(
            f"[Trial {self._trial_count:03d}] "
            f"score={mean_score:.4f} ± {std_score:.4f}  "
            f"best={best_so_far:.4f}  "
            f"α_p={params['alpha_pos']:.2f}  "
            f"α_d={params['alpha_dcm']:.2f}  "
            f"α_E={params['alpha_E']:.2f}  "
            f"α_Q={params['alpha_Q']:.2f}  "
            f"α_M={params['alpha_M']:.2f}  "
            f"α_c={params['alpha_climb']:.2f}  "
            f"β={params['beta']:.2f}  "
            f"lp={params['lp']:.3f}  "
            f"t={elapsed:.1f}s{marker}"
        )

        return mean_score


def _sample_params(trial: optuna.Trial) -> Dict[str, float]:
    """Draw one parameter vector from the search space.

    Args:
        trial:  The Optuna trial object that records the suggestion.

    Returns:
        Dict mapping parameter name → sampled value.
    """
    params: Dict[str, float] = {}
    for name, (dist, lo, hi) in SEARCH_SPACE.items():
        if dist == "loguniform":
            # Sampled uniformly in log-space.  Appropriate when the
            # plausible range spans multiple orders of magnitude.
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        elif dist == "uniform":
            params[name] = trial.suggest_float(name, lo, hi)
        else:
            raise ValueError(f"Unknown distribution '{dist}' for parameter '{name}'.")
    return params


# ==========================================================================
# Phase 3: create study and run
# ==========================================================================

def main() -> None:
    """Main entry point for the tuning script."""

    # ---- Build configuration objects ----
    eval_cfg = EvalConfig(
        num_envs=args_cli.num_envs,
        rollout_steps=args_cli.rollout_steps,
        num_repeat=args_cli.num_repeat,
        device=args_cli.device,
    )
    optuna_cfg = OptunaConfig(
        n_trials=args_cli.n_trials,
        n_startup_trials=args_cli.n_startup_trials,
        seed=args_cli.seed,
        study_name=args_cli.study_name,
        storage=args_cli.storage,
        timeout=args_cli.timeout,
    )

    print("=" * 72)
    print("  DCM Foothold Planner — Bayesian Weight Optimisation")
    print("=" * 72)
    print(f"  Task:       {args_cli.task}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Trials:     {optuna_cfg.n_trials}")
    print(f"  Warmup:     {optuna_cfg.n_startup_trials}")
    print(f"  Envs:       {eval_cfg.num_envs}")
    print(f"  Steps:      {eval_cfg.rollout_steps}")
    print(f"  Repeats:    {eval_cfg.num_repeat}")
    print(f"  Timeout:    {optuna_cfg.timeout // 3600}h" if optuna_cfg.timeout else "  Timeout:    none")
    print(f"  Storage:    {optuna_cfg.storage or 'in-memory'}")
    print("=" * 72)

    # ---- Create the evaluator (expensive: loads environment + policy) ----
    print("\n[INFO] Creating evaluation environment ...")
    evaluator = PlannerEvaluator(
        cfg=eval_cfg,
        task_name=args_cli.task,
        checkpoint_path=args_cli.checkpoint,
    )

    # ---- Quick baseline sanity check ----
    print("[INFO] Running baseline evaluation with default parameters ...")
    baseline_score, baseline_std = evaluator.evaluate_baseline(num_repeat=3)
    print(f"[INFO] Baseline score: {baseline_score:.4f} ± {baseline_std:.4f}")

    # ---- Create the Optuna study ----
    sampler = TPESampler(
        seed=optuna_cfg.seed,
        n_startup_trials=optuna_cfg.n_startup_trials,
        multivariate=True,
        # ``constant_liar=True`` enables parallel trial evaluation by
        # temporarily assuming pending trials have the same objective
        # value as the best observed trial.
        constant_liar=True,
    )

    pruner = MedianPruner(
        n_startup_trials=optuna_cfg.n_startup_trials,
        n_warmup_steps=0,          # our objective reports once per trial
        interval_steps=1,
        n_min_trials=1,
    ) if eval_cfg.enable_pruning else None

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # resume a previous run if storage is persistent
    )

    # ---- Enqueue the default parameters as the first trial ----
    # This guarantees that TPE has a known-good point to start from.
    study.enqueue_trial(DEFAULT_PARAMS)

    # ---- Build the objective function ----
    objective = _Objective(evaluator, eval_cfg)

    # ---- Run ----
    print(f"\n[INFO] Starting optimisation ({optuna_cfg.n_trials} trials) ...\n")
    t_total_start = time.perf_counter()

    try:
        study.optimize(
            objective,
            n_trials=optuna_cfg.n_trials,
            timeout=optuna_cfg.timeout,
            # ``gc_after_trial=True`` helps free GPU memory between trials
            # in case PyTorch caches accumulate.
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Optimisation interrupted by user.  Printing best result so far.")
    except Exception as exc:
        print(f"\n[ERROR] Optimisation aborted: {exc}")
        import traceback
        traceback.print_exc()

    t_total = time.perf_counter() - t_total_start
    print(f"\n[INFO] Total wall-clock time: {t_total / 3600:.2f} h")

    # ---- Print results ----
    _print_results(study, baseline_score)

    # ---- Cleanup ----
    evaluator.close()
    simulation_app.close()


# ==========================================================================
# Results output
# ==========================================================================

def _print_results(study: optuna.Study, baseline_score: float) -> None:
    """Print the best trial and a summary table."""

    if len(study.trials) == 0:
        print("\n[WARN] No trials completed.")
        return

    best = study.best_trial
    print("\n" + "=" * 72)
    print("  OPTIMISATION RESULTS")
    print("=" * 72)
    print(f"  Best trial:       #{best.number}")
    print(f"  Best score:       {best.value:.4f}")
    print(f"  Baseline score:   {baseline_score:.4f}")
    if best.value is not None:
        improvement = (best.value / max(baseline_score, 1e-6) - 1.0) * 100
        print(f"  vs baseline:      {improvement:+.1f}%")
    print()

    # ---- Best parameters ----
    print("  Best parameters:")
    for name in SEARCH_SPACE:
        val = best.params.get(name, "N/A")
        default = DEFAULT_PARAMS.get(name, float("nan"))
        if isinstance(val, float) and default != 0:
            delta_pct = (val / default - 1.0) * 100
            print(f"    {name:16s} = {val:.4f}  (default={default:.4f}, {delta_pct:+.0f}%)")
        else:
            print(f"    {name:16s} = {val}")
    print()

    # ---- Top-5 trials for reference ----
    print("  Top-5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)
    for rank, t in enumerate(completed[:5], start=1):
        score_str = f"{t.value:.4f}" if t.value is not None else "N/A"
        print(f"    #{rank}  Trial {t.number:03d}  score={score_str}")
    print()

    # ---- Parameter importance (if enough trials) ----
    if len(completed) >= 20:
        try:
            importances = optuna.importance.get_param_importances(study)
            print("  Parameter importance (fanova):")
            for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"    {name:16s}  {imp:.3f}  {bar}")
            print()
        except Exception:
            # ``get_param_importances`` may fail with small studies.
            pass

    # ---- Save best params as Python snippet ----
    print("  Best params as Python dict:")
    print("  " + "-" * 68)
    print("  BEST_PARAMS = {")
    for name in SEARCH_SPACE:
        val = best.params.get(name, DEFAULT_PARAMS[name])
        print(f"      \"{name}\": {val:.4f},")
    print("  }")
    print("  " + "-" * 68)
    print()

    # ---- Persist to JSON for later analysis ----
    _save_results_json(study, baseline_score)


def _save_results_json(study: optuna.Study, baseline_score: float) -> None:
    """Write the full trial history to a JSON file for offline analysis."""
    import json
    from datetime import datetime

    out_path = os.path.join(
        os.path.dirname(__file__),
        f"tune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": {
                    k: v for k, v in t.user_attrs.items()
                    if not k.startswith("param_")
                },
            })

    result = {
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_score": study.best_value,
        "baseline_score": baseline_score,
        "best_params": study.best_params,
        "n_trials_completed": len(trials_data),
        "trials": trials_data,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Results saved to {out_path}")


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == "__main__":
    main()
