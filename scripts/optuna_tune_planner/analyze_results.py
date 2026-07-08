#!/usr/bin/env python3
"""
analyze_results.py — offline analysis and visualisation of tuning results.

Usage::

    python scripts/optuna_tune_planner/analyze_results.py \\
        --input tune_results_20260708_120000.json \\
        [--output-dir ./analysis_figs/]

This script reads the JSON file produced by ``tune_planner.py`` and
generates:
  - Optimisation history plot (score vs trial)
  - Parameter-importance bar chart
  - Parallel-coordinate plot (interactive HTML, requires plotly)
  - Pairwise scatter matrix for the top parameters
  - Text report comparing best vs baseline

All plots are saved to ``--output-dir`` (default: same directory as the
input file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Analyse Optuna tuning results for DCM planner weights."
)
parser.add_argument(
    "--input", "-i",
    type=str,
    required=True,
    help="Path to the JSON results file from tune_planner.py.",
)
parser.add_argument(
    "--output-dir", "-o",
    type=str,
    default=None,
    help="Directory for output figures.  Default: same directory as input file.",
)
parser.add_argument(
    "--top-n",
    type=int,
    default=10,
    help="Number of top trials to highlight in the report.",
)
args_cli = parser.parse_args()

# ---- Resolve output directory ----
if args_cli.output_dir is None:
    args_cli.output_dir = os.path.dirname(os.path.abspath(args_cli.input))
os.makedirs(args_cli.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> Dict[str, Any]:
    """Load and validate the JSON results file."""
    with open(path, "r") as f:
        data = json.load(f)

    required_keys = ["trials", "best_params", "best_score", "baseline_score"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in results file.")

    if len(data["trials"]) == 0:
        raise ValueError("No completed trials in the results file.")

    return data


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(data: Dict[str, Any], top_n: int = 10) -> None:
    """Print a detailed text summary to stdout."""
    trials = data["trials"]
    best_score = data["best_score"]
    baseline_score = data["baseline_score"]
    best_params = data["best_params"]

    print("=" * 72)
    print("  DCM PLANNER TUNING — ANALYSIS REPORT")
    print("=" * 72)

    # ---- Summary statistics ----
    scores = [t["value"] for t in trials if t["value"] is not None]
    print(f"\n  Trials completed:     {len(trials)}")
    print(f"  Score range:          [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Score mean ± std:     {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Baseline score:        {baseline_score:.4f}")
    print(f"  Best score:            {best_score:.4f}")
    improvement = (best_score / max(baseline_score, 1e-6) - 1.0) * 100
    print(f"  Improvement:           {improvement:+.1f}%")
    print()

    # ---- Best parameters vs baseline ----
    print("  BEST PARAMETERS:")
    from scripts.optuna_tune_planner.config import DEFAULT_PARAMS
    print(f"  {'Parameter':16s} {'Default':>10s} {'Best':>10s} {'Delta %':>10s}")
    print("  " + "-" * 50)
    for name in DEFAULT_PARAMS:
        default = DEFAULT_PARAMS[name]
        best = best_params.get(name, default)
        delta = (best / default - 1.0) * 100 if default != 0 else 0.0
        print(f"  {name:16s} {default:10.4f} {best:10.4f} {delta:+9.1f}%")
    print()

    # ---- Top-N trials ----
    sorted_trials = sorted(trials, key=lambda t: t["value"] or -float("inf"), reverse=True)
    print(f"  TOP-{top_n} TRIALS:")
    print(f"  {'Rank':5s} {'Trial#':7s} {'Score':>10s}  Parameters")
    print("  " + "-" * 72)
    for rank, t in enumerate(sorted_trials[:top_n], start=1):
        params_str = ", ".join(
            f"{k}={v:.3f}" for k, v in t["params"].items()
        )
        print(f"  {rank:5d} {t['number']:7d} {t['value']:10.4f}  {params_str}")
    print()

    # ---- Correlation analysis ----
    print("  PARAMETER-SCORE CORRELATIONS (Pearson r):")
    param_names = list(trials[0]["params"].keys())
    for name in param_names:
        p_vals = np.array([t["params"].get(name, np.nan) for t in trials])
        valid = ~np.isnan(p_vals)
        if valid.sum() > 5:
            r = np.corrcoef(p_vals[valid], np.array(scores)[valid])[0, 1]
            print(f"    {name:16s}  r = {r:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Matplotlib visualisations
# ---------------------------------------------------------------------------

def plot_optimization_history(
    trials: List[Dict[str, Any]],
    baseline_score: float,
    output_dir: str,
) -> None:
    """Plot score vs trial number with a rolling-best overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping history plot.")
        return

    scores = np.array([t["value"] for t in trials if t["value"] is not None])
    trial_nums = np.arange(1, len(scores) + 1)

    # Rolling best (cumulative maximum).
    rolling_best = np.maximum.accumulate(scores)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trial_nums, scores, "o", alpha=0.4, markersize=3, color="steelblue",
            label="Trial score")
    ax.plot(trial_nums, rolling_best, "-", linewidth=2, color="darkorange",
            label="Best so far")
    ax.axhline(y=baseline_score, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline_score:.3f})")
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("DCM Planner Weight Optimisation — Score vs Trial", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "optimization_history.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_param_importance(
    trials: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart showing the Pearson correlation of each parameter with the
    composite score.  A crude but interpretable importance metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping importance plot.")
        return

    param_names = list(trials[0]["params"].keys())
    scores = np.array([t["value"] for t in trials if t["value"] is not None])

    correlations = {}
    for name in param_names:
        p_vals = np.array([t["params"].get(name, np.nan) for t in trials])
        valid = ~np.isnan(p_vals)
        if valid.sum() > 5:
            correlations[name] = abs(np.corrcoef(p_vals[valid], scores[valid])[0, 1])
        else:
            correlations[name] = 0.0

    # Sort by absolute correlation.
    sorted_items = sorted(correlations.items(), key=lambda x: -x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlabel("|Pearson r| with Composite Score", fontsize=12)
    ax.set_title("Parameter Importance (Linear Correlation)", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate bars.
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "param_importance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_pairwise(
    trials: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Pairwise scatter matrix for the top 4 most-correlated parameters.

    Each point is a trial, coloured by its composite score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping pairwise plot.")
        return

    param_names = list(trials[0]["params"].keys())
    scores = np.array([t["value"] for t in trials if t["value"] is not None])

    # Pick top 4 parameters by correlation.
    correlations = {}
    for name in param_names:
        p_vals = np.array([t["params"].get(name, np.nan) for t in trials])
        valid = ~np.isnan(p_vals)
        if valid.sum() > 5:
            correlations[name] = abs(np.corrcoef(p_vals[valid], scores[valid])[0, 1])

    top_params = sorted(correlations, key=lambda k: -correlations[k])[:4]
    if len(top_params) < 2:
        print("[WARN] Not enough parameters for pairwise plot.")
        return

    n = len(top_params)
    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3))
    param_arrays = {name: np.array([t["params"][name] for t in trials]) for name in top_params}

    for i, px in enumerate(top_params):
        for j, py in enumerate(top_params):
            ax = axes[i][j] if n > 1 else axes
            if i == j:
                # Diagonal: histogram.
                ax.hist(param_arrays[px], bins=30, color="steelblue", alpha=0.7)
                ax.set_xlabel(px, fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
            else:
                sc = ax.scatter(
                    param_arrays[py], param_arrays[px],
                    c=scores, cmap="viridis", alpha=0.5, s=10,
                )
                ax.set_xlabel(py, fontsize=9)
                ax.set_ylabel(px, fontsize=9)

    fig.suptitle("Pairwise Parameter Scatter (coloured by score)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "pairwise_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_parallel_coordinate(
    trials: List[Dict[str, Any]],
    best_params: Dict[str, float],
    output_dir: str,
) -> None:
    """Interactive parallel-coordinate plot using plotly (optional)."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("[WARN] plotly not available — skipping parallel-coordinate plot.")
        print("       Install with: pip install plotly")
        return

    param_names = list(best_params.keys())
    sorted_trials = sorted(trials, key=lambda t: t["value"] or -float("inf"))

    # Build one line per trial.
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=[t["value"] for t in sorted_trials],
                colorscale="Viridis",
                showscale=True,
                cmin=min(t["value"] for t in sorted_trials if t["value"] is not None),
                cmax=max(t["value"] for t in sorted_trials if t["value"] is not None),
            ),
            dimensions=[
                dict(
                    label=name,
                    values=[t["params"][name] for t in sorted_trials],
                )
                for name in param_names
            ],
        )
    )
    fig.update_layout(
        title="Parallel Coordinate Plot — DCM Planner Tuning",
        width=1200,
        height=600,
    )
    out_path = os.path.join(output_dir, "parallel_coordinate.html")
    pio.write_html(fig, out_path)
    print(f"[INFO] Saved {out_path}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    data = load_results(args_cli.input)

    # Text report.
    print_report(data, top_n=args_cli.top_n)

    # Visualisations.
    trials = data["trials"]
    baseline = data["baseline_score"]
    best_params = data["best_params"]
    out_dir = args_cli.output_dir

    plot_optimization_history(trials, baseline, out_dir)
    plot_param_importance(trials, out_dir)
    plot_pairwise(trials, out_dir)
    plot_parallel_coordinate(trials, best_params, out_dir)

    print(f"\n[INFO] All analyses saved to {out_dir}/")


if __name__ == "__main__":
    main()
