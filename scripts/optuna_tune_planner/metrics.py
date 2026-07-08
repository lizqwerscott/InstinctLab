"""
metrics.py — incremental metric collection and composite scoring.

A ``MetricsAccumulator`` collects per-step statistics during a rollout
and, at the end, computes a single scalar score that Optuna maximises.

The composite score blends four sub-metrics, each normalised to ~[0, 1]
so that the mixing weights represent relative importance rather than
compensating for different numerical scales.

Normalisation strategy:
  - Rewards and success rates naturally live in [0, 1].
  - Tracking error and slip distance are clipped to a plausible maximum
    and then mapped to [0, 1] via ``1.0 - clip(value / max_value, 0, 1)``,
    turning them into "scores" where higher is always better.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from scripts.optuna_tune_planner.config import EvalConfig


# ---------------------------------------------------------------------------
# Per-step bucket helpers
# ---------------------------------------------------------------------------

@dataclass
class _StepBucket:
    """Accumulates scalar statistics for one terrain type using Welford's
    online algorithm (numerically stable mean and variance)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0          # sum of squared differences from mean
    values: List[float] = field(default_factory=list)  # kept for percentile queries

    def add(self, value: float) -> None:
        """Add one observation.  O(1) per call."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.values.append(value)

    @property
    def variance(self) -> float:
        """Sample variance (unbiased estimator)."""
        return self.m2 / max(1, self.count - 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


# ---------------------------------------------------------------------------
# Main accumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """Collects per-step telemetry and computes a composite score.

    Designed to be called inside the rollout loop::

        accum = MetricsAccumulator(cfg)
        for step in range(rollout_steps):
            obs, rew, dones, infos = env.step(actions)
            accum.update(
                foothold_reward=...,
                tracking_error=...,
                foot_slip=...,
                done_mask=dones,
                terrain_ids=terrain_ids,
            )
        final_score = accum.compute_score()

    Sub-metrics are tracked both globally and per-terrain-type so the
    composite score can weight each terrain according to ``cfg.terrain_weights``.
    """

    # ---- Clipping thresholds for penalty-to-score normalisation ----
    # These are upper bounds on what a "reasonable" policy produces.
    # Values exceeding the threshold are clipped; this prevents a single
    # outlier from dominating the composite score.

    _MAX_TRACKING_ERROR: float = 2.0   # m/s — velocity tracking L2 norm
    _MAX_FOOT_SLIP: float = 0.1        # m — slip distance per step

    def __init__(self, cfg: EvalConfig) -> None:
        """Create an empty accumulator.

        Args:
            cfg:  Evaluation configuration (terrain weights, mixing weights).
        """
        self._cfg = cfg

        # ---- Global (all-terrain) accumulators ----
        self.global_foothold_reward = _StepBucket()
        self.global_tracking_error = _StepBucket()
        self.global_foot_slip = _StepBucket()

        # ---- Per-terrain accumulators ----
        # Keyed by terrain sub-type name (e.g. "pyramid_stairs").
        self._per_terrain: Dict[str, Dict[str, _StepBucket]] = defaultdict(
            lambda: {
                "foothold_reward": _StepBucket(),
                "tracking_error": _StepBucket(),
                "foot_slip": _StepBucket(),
            }
        )

        # ---- Termination tracking ----
        self._total_envs: int = 0
        self._terminated_envs: int = 0
        self._per_terrain_total: Dict[str, int] = defaultdict(int)
        self._per_terrain_terminated: Dict[str, int] = defaultdict(int)

        # ---- Step counter ----
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        foothold_reward: np.ndarray | List[float],
        tracking_error: np.ndarray | List[float],
        foot_slip: np.ndarray | List[float],
        done_mask: np.ndarray | List[bool],
        terrain_ids: Optional[np.ndarray | List[int]] = None,
    ) -> None:
        """Ingest one step of rollout data.

        Args:
            foothold_reward:  Per-environment foothold proximity reward at
                this step.  Shape ``(num_envs,)``.
            tracking_error:  Per-environment L2 velocity tracking error.
            foot_slip:  Per-environment foot-slip distance in meters.
            done_mask:  Boolean mask indicating which environments just
                terminated at this step.
            terrain_ids:  Integer terrain-type index per environment, with
                the same length as the other arrays.  If ``None``, all
                environments are treated as a single "global" terrain.
        """
        self._step_count += 1
        n_envs = len(foothold_reward)

        # Convert to numpy arrays if needed.
        if not isinstance(foothold_reward, np.ndarray):
            foothold_reward = np.asarray(foothold_reward)
        if not isinstance(tracking_error, np.ndarray):
            tracking_error = np.asarray(tracking_error)
        if not isinstance(foot_slip, np.ndarray):
            foot_slip = np.asarray(foot_slip)
        if not isinstance(done_mask, np.ndarray):
            done_mask = np.asarray(done_mask)

        # Filter out NaN values (can occur when a sensor misses).
        valid = (
            ~np.isnan(foothold_reward)
            & ~np.isnan(tracking_error)
            & ~np.isnan(foot_slip)
        )

        # ---- Global accumulation ----
        for arr, bucket in [
            (foothold_reward, self.global_foothold_reward),
            (tracking_error, self.global_tracking_error),
            (foot_slip, self.global_foot_slip),
        ]:
            for v in arr[valid]:
                bucket.add(float(v))

        # ---- Per-terrain accumulation ----
        if terrain_ids is not None:
            terrain_ids = np.asarray(terrain_ids)
            # Map terrain index → terrain name via the environment's
            # terrain sub-type mapping (set externally before rollout).
            for tid in np.unique(terrain_ids[valid]):
                mask = valid & (terrain_ids == tid)
                tname = self._resolve_terrain_name(tid)
                self._per_terrain[tname]["foothold_reward"].add(
                    float(np.mean(foothold_reward[mask]))
                )
                self._per_terrain[tname]["tracking_error"].add(
                    float(np.mean(tracking_error[mask]))
                )
                self._per_terrain[tname]["foot_slip"].add(
                    float(np.mean(foot_slip[mask]))
                )

        # ---- Termination counting ----
        self._total_envs += n_envs
        self._terminated_envs += int(done_mask.sum())
        if terrain_ids is not None:
            for tid in np.unique(terrain_ids):
                tname = self._resolve_terrain_name(tid)
                mask = terrain_ids == tid
                self._per_terrain_total[tname] += int(mask.sum())
                self._per_terrain_terminated[tname] += int(
                    done_mask[mask].sum()
                )

    def compute_score(self) -> float:
        """Compute the weighted composite score.

        All sub-scores are in the range [0, 1] where higher is better.
        The mixing weights come from ``EvalConfig``.

        Returns:
            A single scalar.  Larger → better planner parameters.
        """
        # ---- 1. Foothold proximity score (already in [0, 1]) ----
        s_foothold = max(0.0, self.global_foothold_reward.mean)

        # ---- 2. Success rate score ----
        if self._total_envs > 0:
            s_success = 1.0 - self._terminated_envs / self._total_envs
        else:
            s_success = 0.0

        # ---- 3. Tracking quality score ----
        # Penalty → score:  perfect tracking (0 error) → 1.0
        raw_tracking = self.global_tracking_error.mean
        s_tracking = 1.0 - min(raw_tracking / self._MAX_TRACKING_ERROR, 1.0)

        # ---- 4. Foot-slip quality score ----
        raw_slip = self.global_foot_slip.mean
        s_slip = 1.0 - min(raw_slip / self._MAX_FOOT_SLIP, 1.0)

        # ---- 5. Per-terrain composite (if terrain data available) ----
        terrain_scores: Dict[str, float] = {}
        for tname in self._cfg.terrain_weights:
            if tname in self._per_terrain:
                bt = self._per_terrain[tname]
                t_foothold = bt["foothold_reward"].mean
                t_tracking = 1.0 - min(
                    bt["tracking_error"].mean / self._MAX_TRACKING_ERROR, 1.0
                )
                t_slip = 1.0 - min(
                    bt["foot_slip"].mean / self._MAX_FOOT_SLIP, 1.0
                )
                # Success rate per terrain
                total = self._per_terrain_total.get(tname, 1)
                term = self._per_terrain_terminated.get(tname, 0)
                t_success = 1.0 - term / max(total, 1)

                terrain_scores[tname] = (
                    self._cfg.w_foothold_proximity * t_foothold
                    + self._cfg.w_success_rate * t_success
                    + self._cfg.w_tracking_penalty * t_tracking
                    + self._cfg.w_foot_slip_penalty * t_slip
                )

        if terrain_scores:
            # Weighted sum over terrains
            s_terrain = sum(
                self._cfg.terrain_weights.get(name, 0.0) * score
                for name, score in terrain_scores.items()
            )
            # Normalise by the sum of weights for terrains we actually saw
            active_weights = sum(
                self._cfg.terrain_weights.get(name, 0.0)
                for name in terrain_scores
            )
            s_terrain = s_terrain / max(active_weights, 1e-6)
        else:
            s_terrain = 0.0

        # ---- 6. Global composite (fallback when per-terrain unavailable) ----
        s_global = (
            self._cfg.w_foothold_proximity * s_foothold
            + self._cfg.w_success_rate * s_success
            + self._cfg.w_tracking_penalty * s_tracking
            + self._cfg.w_foot_slip_penalty * s_slip
        )

        # Blend: if we have terrain data, use it; otherwise fall back to global.
        if terrain_scores:
            return 0.7 * s_terrain + 0.3 * s_global
        else:
            return s_global

    # ------------------------------------------------------------------
    # Terrain-name resolution
    # ------------------------------------------------------------------

    # Class-level mapping from terrain index to sub-terrain name.
    # Populated externally by the evaluator before the first rollout,
    # because the mapping depends on the terrain generator config.
    _terrain_id_to_name: Dict[int, str] = {}

    @classmethod
    def set_terrain_mapping(cls, mapping: Dict[int, str]) -> None:
        """Register terrain-index → name mapping (called once at setup)."""
        cls._terrain_id_to_name = mapping

    @staticmethod
    def _resolve_terrain_name(terrain_id: int) -> str:
        """Convert a terrain-type integer index to a human-readable name."""
        return MetricsAccumulator._terrain_id_to_name.get(
            terrain_id, f"terrain_{terrain_id}"
        )

    # ------------------------------------------------------------------
    # Query helpers (for logging / analysis)
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Return a flat dict of key metrics for logging."""
        return {
            "foothold_reward_mean": self.global_foothold_reward.mean,
            "foothold_reward_std": self.global_foothold_reward.std,
            "success_rate": (
                1.0 - self._terminated_envs / max(self._total_envs, 1)
            ),
            "tracking_error_mean": self.global_tracking_error.mean,
            "foot_slip_mean": self.global_foot_slip.mean,
            "composite_score": self.compute_score(),
            "step_count": self._step_count,
        }
