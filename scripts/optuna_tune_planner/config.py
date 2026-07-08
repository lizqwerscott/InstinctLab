"""
config.py — search-space bounds, evaluation hyper-parameters, and terrain weights.

All tunable knobs are gathered here so that the other modules are pure logic.
Each constant / dataclass field carries a comment explaining its effect on the
optimisation loop.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# Each entry: (distribution, low, high)
#   "loguniform" — sampled uniformly in log-space; appropriate when the
#                  plausible range spans multiple orders of magnitude.
#   "uniform"    — sampled uniformly in linear space.

SEARCH_SPACE: Dict[str, Tuple[str, float, float]] = {
    # ---- Cost-channel weights (Eq. 1 in TACT-ful paper) ----
    # Position residual: squared distance between grid cell and nominal
    # step location (L_nom, W_nom).  A larger value forces the planner to
    # stay close to the nominal foot placement.
    "alpha_pos": ("loguniform", 0.1, 10.0),

    # DCM residual: penalises deviation from the divergent-component-of-
    # motion target.  Larger → more emphasis on dynamic stability.
    "alpha_dcm": ("loguniform", 0.1, 5.0),

    # Steepness (Sobel gradient magnitude of the heightmap).  Larger →
    # the planner avoids slanted terrain more aggressively.
    "alpha_E": ("loguniform", 0.1, 5.0),

    # Flatness (max - min height inside the footprint kernel).  Larger →
    # smoother landing patches are preferred.
    "alpha_Q": ("loguniform", 1.0, 50.0),

    # Feasibility penalty for vertical step height exceeding the velocity-
    # dependent threshold h_eff.
    "alpha_M": ("loguniform", 0.5, 30.0),

    # Climb bonus for ascending steps when forward velocity is high.
    # Subtracted from total cost → *negative* contribution.  Larger →
    # the planner is more willing to climb stairs.
    "alpha_climb": ("loguniform", 0.1, 10.0),

    # ---- Lateral-vs-forward trade-off (inside d_pos) ----
    # β multiplies the lateral component of the position residual:
    #   d_pos = (gx - L_nom)² + β·(gy - W_nom)²
    # Larger β → the planner cares more about keeping the correct step
    # width.
    "beta": ("loguniform", 0.5, 10.0),

    # ---- Pelvis half-width (m) ----
    # Physical parameter controlling the nominal lateral step offset.
    # Constrained to anatomically plausible values for the G1 robot.
    # TACT-ful default 0.20; smaller → narrower stance, larger → wider.
    "lp": ("uniform", 0.10, 0.35),
}

# ---------------------------------------------------------------------------
# Default parameter values (current production settings)
# ---------------------------------------------------------------------------
# Used as the first enqueued trial so TPE starts from a known-good point.
DEFAULT_PARAMS: Dict[str, float] = {
    "alpha_pos": 1.0,
    "alpha_dcm": 0.5,
    "alpha_E": 0.6,
    "alpha_Q": 10.0,
    "alpha_M": 6.0,
    "alpha_climb": 1.5,
    "beta": 2.5,
    "lp": 0.20,
}

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Hyper-parameters controlling the cost and reliability of each trial."""

    # ---- Simulation scale ----
    # Number of parallel environments used during evaluation.
    # Lower than training (4096) to keep rollout fast; high enough to get
    # statistically meaningful terrain coverage.
    num_envs: int = 512

    # Number of simulation steps per rollout.  500 steps @ dt=0.02 s =
    # 10 s of simulated time, enough to traverse ~5 m at 0.5 m/s.
    rollout_steps: int = 500

    # Each parameter set is evaluated num_repeat times with different
    # random seeds.  The mean score is returned to Optuna.  Higher →
    # less noise but more compute.
    num_repeat: int = 3

    # ---- Terrain weighting ----
    # Maps terrain sub-type names (as defined in ROUGH_TERRAINS_CFG) to
    # the fraction of the total score they contribute.  Must sum to ~1.
    terrain_weights: Dict[str, float] = field(default_factory=lambda: {
        "perlin_rough":       0.25,   # continuous rough ground
        "pyramid_stairs":      0.20,   # ascending stairs
        "pyramid_stairs_inv": 0.15,   # descending stairs
        "up_down":            0.20,   # stairs up then down
        "down_up":            0.20,   # stairs down then up
    })

    # ---- Composite-score mixing weights ----
    # final_score = Σ w_i · metric_i  (all metrics normalised to [0, 1]
    # so the weights are relative importance, not absolute scaling).

    # Planner quality: how close the actual foot placement is to the
    # planned foothold at touchdown.
    w_foothold_proximity: float = 0.35

    # Task completion: fraction of environments that survive the full
    # rollout without a termination (fall, boundary, orientation).
    w_success_rate: float = 0.35

    # Task performance: L2 error between commanded and actual velocity.
    # Penalty sign is negative — a large tracking error *reduces* the
    # composite score.
    w_tracking_penalty: float = 0.15

    # Contact quality: distance the foot slides after touchdown.
    # Penalty sign is negative.
    w_foot_slip_penalty: float = 0.15

    # ---- Pruning ----
    # If enabled, the first prune_quick_steps of the rollout are used as
    # a "quick filter" on the simplest terrain (perlin_rough).  Trials
    # that underperform the running median by this factor are pruned
    # before the expensive full-terrain evaluation.
    enable_pruning: bool = True
    prune_quick_steps: int = 100          # steps for the quick-filter phase
    prune_percentile: float = 30.0        # prune if score < P30 of completed trials

    # ---- Hardware ----
    # Isaac Sim device string, e.g. "cuda:0" or "cpu".
    device: str = "cuda:0"

@dataclass
class OptunaConfig:
    """Optuna study-level configuration."""

    # Total number of trials (including warmup).
    n_trials: int = 100

    # Number of initial random-sampler trials before TPE takes over.
    # Larger → better exploration; smaller → faster exploitation.
    n_startup_trials: int = 15

    # Random seed for reproducibility of the TPE sampler.
    seed: int = 42

    # Study name (appears in optuna-dashboard and log files).
    study_name: str = "dcm_planner_tune"

    # Storage URL.  None → in-memory (lost on exit).
    # "sqlite:///tune_planner.db" → persists across restarts.
    storage: Optional[str] = None

    # Timeout in seconds.  None → no timeout.
    timeout: Optional[int] = 8 * 3600  # 8 hours


# ---------------------------------------------------------------------------
# Final validation configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Post-optimisation validation: short RL training for top-N candidates."""

    # Number of top trials to validate with real training.
    top_n: int = 3

    # Training iterations for each validation run.
    # Shorter than full training (30000) but long enough to see trends.
    validation_iterations: int = 5000

    # Number of seeds per candidate.
    num_seeds: int = 3
