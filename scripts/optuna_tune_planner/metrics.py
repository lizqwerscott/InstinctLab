"""
metrics.py — incremental collection of ankle-corrected foothold proximity.

The ``MetricsAccumulator`` is called after every simulation step during a
rollout.  At the end, ``compute_score()`` returns the mean foothold
proximity, weighted by terrain type.  This is the scalar that Optuna
maximises.

Score definition
----------------
  score = mean( exp(-sigma_p * ankle_corrected_distance²) )

where ``ankle_corrected_distance`` is the L2 distance between the actual
ankle position and the planned foothold target, with a 3.5 cm backward
offset applied to the target (ankle sits behind the foot centre on G1).

The score is computed both globally and per-terrain, then blended using
``EvalConfig.terrain_weights``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from scripts.optuna_tune_planner.config import EvalConfig


# ---------------------------------------------------------------------------
# Per-step bucket (Welford's online mean)
# ---------------------------------------------------------------------------

@dataclass
class _StepBucket:
    """Accumulates scalar statistics using Welford's online algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return np.sqrt(self.m2 / max(1, self.count - 1))


# ---------------------------------------------------------------------------
# Main accumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """Collects per-step foothold proximity and computes a terrain-weighted
    mean score.

    Usage inside the rollout loop::

        accum = MetricsAccumulator(cfg)
        for step in range(rollout_steps):
            obs, rew, dones, infos = env.step(actions)
            foothold_score = compute_ankle_corrected_score(...)
            accum.update(foothold_score, terrain_ids)
        final_score = accum.compute_score()
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self._cfg = cfg
        self.global_foothold = _StepBucket()
        self._per_terrain: Dict[str, _StepBucket] = defaultdict(_StepBucket)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        foothold_score: np.ndarray | List[float],
        terrain_ids: Optional[np.ndarray | List[int]] = None,
    ) -> None:
        """Ingest one step of rollout data.

        Args:
            foothold_score:  Per-environment ankle-corrected foothold
                proximity, shape ``(num_envs,)``, each value in [0, 1].
            terrain_ids:  Optional integer terrain-type index per env.
        """
        self._step_count += 1

        if not isinstance(foothold_score, np.ndarray):
            foothold_score = np.asarray(foothold_score)
        valid = ~np.isnan(foothold_score)

        # Global
        for v in foothold_score[valid]:
            self.global_foothold.add(float(v))

        # Per-terrain
        if terrain_ids is not None:
            terrain_ids = np.asarray(terrain_ids)
            for tid in np.unique(terrain_ids[valid]):
                mask = valid & (terrain_ids == tid)
                tname = self._resolve_terrain_name(tid)
                self._per_terrain[tname].add(
                    float(np.mean(foothold_score[mask]))
                )

    # ------------------------------------------------------------------
    def compute_score(self) -> float:
        """Terrain-weighted mean foothold proximity.

        Returns a scalar in [0, 1]; higher is better.
        """
        terrain_scores: Dict[str, float] = {}
        for tname, bucket in self._per_terrain.items():
            terrain_scores[tname] = bucket.mean

        if terrain_scores:
            score = sum(
                self._cfg.terrain_weights.get(name, 0.0) * s
                for name, s in terrain_scores.items()
            )
            active_weight = sum(
                self._cfg.terrain_weights.get(name, 0.0)
                for name in terrain_scores
            )
            score /= max(active_weight, 1e-6)
            return score
        # Fallback: global mean
        return max(0.0, self.global_foothold.mean)

    # ------------------------------------------------------------------
    # Terrain-name resolution (class-level mapping, set once at startup)
    # ------------------------------------------------------------------
    _terrain_id_to_name: Dict[int, str] = {}

    @classmethod
    def set_terrain_mapping(cls, mapping: Dict[int, str]) -> None:
        cls._terrain_id_to_name = mapping

    @staticmethod
    def _resolve_terrain_name(tid: int) -> str:
        return MetricsAccumulator._terrain_id_to_name.get(
            tid, f"terrain_{tid}"
        )

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        return {
            "foothold_mean": self.global_foothold.mean,
            "foothold_std": self.global_foothold.std,
            "composite_score": self.compute_score(),
            "step_count": self._step_count,
        }
