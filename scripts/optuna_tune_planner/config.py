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
    "alpha_pos":   ("loguniform", 0.1, 10.0),
    "alpha_dcm":   ("loguniform", 0.1, 5.0),
    "alpha_E":     ("loguniform", 0.1, 5.0),
    "alpha_Q":     ("loguniform", 1.0, 50.0),
    "alpha_M":     ("loguniform", 0.5, 30.0),
    "alpha_climb": ("loguniform", 0.1, 10.0),
    "beta":        ("loguniform", 0.5, 10.0),

    # ---- Swing-phase duration (s) ----
    # T controls the LIPM time horizon: the planner assumes the swing leg
    # has T seconds to reach the target.  Longer T → larger step length
    # and more DCM amplification; shorter T → more conservative.
    # Range 0.0–0.6 s covers typical humanoid swing durations.
    "T": ("uniform", 0.0, 0.6),
}

# ---------------------------------------------------------------------------
# Fixed parameters (not searched)
# ---------------------------------------------------------------------------
# lp = 0.20 m — pelvis half-width, G1 physical constant.
# k  = 0.0   — slope.  Zero means the DCM planner always assumes flat
#               ground, i.e. the LIPM height z₀ is treated as constant
#               during the swing phase.

# ---------------------------------------------------------------------------
# Default parameter values (current production settings)
# ---------------------------------------------------------------------------
# Used as the first enqueued trial so TPE starts from a known-good point.
DEFAULT_PARAMS: Dict[str, float] = {
    "alpha_pos":   1.0,
    "alpha_dcm":   0.5,
    "alpha_E":     0.6,
    "alpha_Q":     10.0,
    "alpha_M":     6.0,
    "alpha_climb": 1.5,
    "beta":        2.5,
    "T":           0.45,
    # lp is fixed externally, not in this dict.
}

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Hyper-parameters controlling the cost and reliability of each trial."""

    # ---- Simulation scale ----
    num_envs: int = 512
    rollout_steps: int = 500
    num_repeat: int = 3

    # ---- Terrain weighting ----
    # Only stair-type terrains are weighted because foothold_proximity
    # reward is gated to these terrain types (perlin_rough always returns
    # zero reward, providing no planner-quality signal).
    terrain_weights: Dict[str, float] = field(default_factory=lambda: {
        "pyramid_stairs":      0.25,
        "pyramid_stairs_inv": 0.20,
        "up_down":            0.30,
        "down_up":            0.25,
    })

    # ---- Score definition ----
    # The composite score is simply the mean ankle-corrected foothold
    # proximity across all environments and steps.  No additional task
    # metrics (tracking, slip, success rate) are mixed in because:
    #   - If the planner picks a bad foothold the robot stumbles → low
    #     foothold reward naturally.
    #   - If the robot falls, no foothold reward is produced at all →
    #     score = 0.

    # Gaussian sharpness for converting L2 distance to [0, 1] score.
    # score = exp(-sigma_p * dist²) where dist is the ankle-corrected
    # distance between actual foot placement and planned target.
    sigma_p: float = 10.0

    # ---- Ankle offset ----
    # The DCM planner outputs a target for the *foot centre*.  However the
    # actual landing position is determined by the ankle joint, which on
    # the G1 robot sits ~3.5 cm behind the foot centre (in the robot's
    # forward direction).  We subtract this offset from the planned target
    # before computing the distance.
    ankle_offset: float = 0.035  # metres

    # ---- Pruning ----
    enable_pruning: bool = True
    prune_quick_steps: int = 100
    prune_percentile: float = 30.0

    # ---- Hardware ----
    device: str = "cuda:0"


@dataclass
class OptunaConfig:
    """Optuna study-level configuration."""

    n_trials: int = 100
    n_startup_trials: int = 15
    seed: int = 42
    study_name: str = "dcm_planner_tune"
    storage: Optional[str] = None
    timeout: Optional[int] = 8 * 3600  # 8 hours


# ---------------------------------------------------------------------------
# Final validation configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Post-optimisation validation: short RL training for top-N candidates."""

    top_n: int = 3
    validation_iterations: int = 5000
    num_seeds: int = 3
