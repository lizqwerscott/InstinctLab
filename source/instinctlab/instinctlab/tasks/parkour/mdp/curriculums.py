from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


class foothold_weight_schedule(ManagerTermBase):
    """Curriculum that gates the combined foothold reward's total weight
    (RewTerm.weight) behind velocity tracking performance.

    The term runs in two phases:

    1. **Gate phase** (before the latch): the weight linearly ramps from
       ``start_weight`` toward ``end_weight`` based on the population's
       average instantaneous velocity tracking score, smoothed by an EMA
       that persists across episode boundaries. Tracking is still being
       learned here, so the weight can move up *and* down with the score.

    2. **Latched phase** (once ``EMA >= latch_threshold``): the weight
       decouples from velocity entirely. The latch is one-way — a
       momentary tracking dip can no longer pull the weight back down —
       and the weight then grows monotonically by ``ramp_rate`` **per env
       step** until it reaches ``end_weight``. The ramp is driven by
       ``env.common_step_counter`` (not the call frequency), so its
       timing is exact regardless of how often this term is invoked.
       This removes the oscillation a pure velocity-gate suffers from
       once the robot can track well.

    ``latch_threshold`` defaults to ``vel_tracking_target``. To get a
    visible self-paced climb after latching, keep ``latch_threshold``
    *below* ``vel_tracking_target`` (e.g. latch at 0.6 with target 0.8);
    otherwise the gate has already saturated by the time the latch fires
    and the weight simply holds at ``end_weight``.

    The default ``end_weight=None`` picks up whatever weight is set in
    the reward config, so the curriculum stays in sync.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_term_name = cfg.params.get("reward_term_name", "foothold")
        self._command_name = cfg.params.get("command_name", "base_velocity")
        self._term_cfg = None
        self._initial_weight = None
        self._vel_tracking_ema = 0.0
        self._latched = False
        self._latch_step = 0
        self._latch_base_weight = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "foothold",
        start_weight: float = 0.0,
        end_weight: float | None = None,
        vel_tracking_threshold: float = 0.3,
        vel_tracking_target: float = 0.8,
        vel_ema_alpha: float = 0.005,
        latch_threshold: float | None = None,
        ramp_rate: float = 0.02,
    ) -> dict:
        if self._term_cfg is None:
            try:
                self._term_cfg = env.reward_manager.get_term_cfg(self._reward_term_name)
                self._initial_weight = self._term_cfg.weight
            except ValueError:
                pass

        if self._term_cfg is None:
            return {
                "weight": start_weight,
                "vel_factor": 1.0,
                "vel_tracking_ema": self._vel_tracking_ema,
                "instant_tracking": 0.0,
                "latched": 0.0,
                "ramp_weight": 0.0,
            }

        if end_weight is None:
            end_weight = self._initial_weight

        if latch_threshold is None:
            latch_threshold = vel_tracking_target
        ramp_rate = max(ramp_rate, 0.0)

        # Velocity-tracking gate: scale the weight by the population's
        # instantaneous tracking quality (EMA-smoothed across episodes).
        try:
            command = env.command_manager.get_term(self._command_name)
            robot = command.robot
            lin_vel_error = torch.sum(
                torch.square(command.vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2]),
                dim=1,
            )
            instant_tracking = torch.exp(-lin_vel_error / (command.cfg.lin_vel_metrics_std**2)).mean().item()
        except (ValueError, KeyError, AttributeError):
            instant_tracking = 0.0

        self._vel_tracking_ema = vel_ema_alpha * instant_tracking + (1.0 - vel_ema_alpha) * self._vel_tracking_ema

        if not self._latched:
            # --- gate phase: weight follows (and may oscillate with) tracking ---
            if (
                self._vel_tracking_ema < vel_tracking_threshold
                or vel_tracking_target <= vel_tracking_threshold
            ):
                vel_factor = 0.0 if self._vel_tracking_ema < vel_tracking_threshold else 1.0
            else:
                vel_factor = min(
                    (self._vel_tracking_ema - vel_tracking_threshold)
                    / (vel_tracking_target - vel_tracking_threshold),
                    1.0,
                )
            new_weight = start_weight + (end_weight - start_weight) * vel_factor
            ramp_weight = 0.0

            # one-way latch: once tracking has proven good enough, decouple.
            if self._vel_tracking_ema >= latch_threshold:
                self._latched = True
                # start the self-ramp from the gate value at the latch point,
                # capped so a tracking spike can't jump the ramp ahead.
                if vel_tracking_target > vel_tracking_threshold:
                    latch_cap = (latch_threshold - vel_tracking_threshold) / (
                        vel_tracking_target - vel_tracking_threshold
                    )
                    latch_cap = min(max(latch_cap, 0.0), 1.0)
                else:
                    latch_cap = 1.0 if latch_threshold >= vel_tracking_threshold else 0.0
                self._latch_base_weight = start_weight + (end_weight - start_weight) * min(vel_factor, latch_cap)
                self._latch_step = env.common_step_counter
                # emit the ramp start right away so the latch step doesn't jump
                # to the (uncapped) gate value and drop on the next call
                new_weight = self._latch_base_weight
                ramp_weight = new_weight
        else:
            # --- latched phase: self-paced monotonic climb per env step, decoupled
            # from velocity. Driven by env.common_step_counter so the ramp timing
            # is exact no matter how often this term is invoked.
            steps_since_latch = env.common_step_counter - self._latch_step
            ramp_weight = min(self._latch_base_weight + ramp_rate * steps_since_latch, end_weight)
            new_weight = ramp_weight
            if end_weight > start_weight:
                vel_factor = min((ramp_weight - start_weight) / (end_weight - start_weight), 1.0)
            else:
                vel_factor = 1.0

        self._term_cfg.weight = new_weight
        return {
            "weight": new_weight,
            "vel_factor": vel_factor,
            "vel_tracking_ema": self._vel_tracking_ema,
            "instant_tracking": instant_tracking,
            "latched": float(self._latched),
            "ramp_weight": ramp_weight,
        }
