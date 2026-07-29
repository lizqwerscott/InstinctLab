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
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (
        tracking_exp_vel_yaw > ang_vel_threshold[1]
    )
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


class foothold_proximity_weight_schedule(ManagerTermBase):
    """Curriculum that gates the foothold_proximity reward weight behind
    single-stance gait detection.

    The weight stays at ``start_weight`` until the robot population has
    accumulated enough ``cumulative_single_stance_frames`` (measured by
    ``contact_left XOR contact_right``), which filters out two-foot hopping
    and shuffling.  Once the threshold is crossed, the weight linearly ramps
    up to ``end_weight`` over ``ramp_single_stance_frames`` additional frames.

    Thresholds are expressed per-environment so they work regardless of
    the number of parallel environments:

    - ``min_single_stance_frames`` (default 100,000) — per-env single-stance
      frames before the ramp begins.
    - ``ramp_single_stance_frames`` (default 200,000) — additional per-env
      single-stance frames over which the weight linearly reaches
      ``end_weight``.

    A ``common_step_counter``-based fallback (``safety_steps_per_env``,
    default 1,000,000 steps per env) prevents the weight from remaining at
    zero after a training resume, where the in-memory accumulator is reset.

    Additionally, a velocity-tracking gate (``vel_tracking_threshold`` /
    ``vel_tracking_target``) scales down the weight when the population's
    average ``tracking_exp_vel_xy`` is poor, ensuring the weight only
    ramps up once the robot can actually track commanded velocities.

    The default ``end_weight=None`` picks up whatever weight is set in the
    config, so the curriculum stays in sync if the config changes.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_term_name = cfg.params.get("reward_term_name", "foothold_proximity")
        self._term_cfg = None
        self._initial_weight = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "foothold_proximity",
        start_weight: float = 0.0,
        end_weight: float | None = None,
        min_single_stance_frames: int = 100_000,
        ramp_single_stance_frames: int = 200_000,
        safety_steps_per_env: int = 1_000_000,
        vel_tracking_threshold: float = 0.3,
        vel_tracking_target: float = 0.8,
    ) -> dict:
        if self._term_cfg is None:
            try:
                self._term_cfg = env.reward_manager.get_term_cfg(self._reward_term_name)
                self._initial_weight = self._term_cfg.weight
            except ValueError:
                pass

        if self._term_cfg is None:
            return {"weight": start_weight, "xor_progress": 0.0, "step_progress": 0.0, "vel_factor": 1.0, "base_progress": 0.0, "progress": 0.0, "tracking_score": 0.0}

        if end_weight is None:
            end_weight = self._initial_weight

        num_envs = env.num_envs
        total = self._term_cfg.func.cumulative_single_stance_frames
        total_per_env = total / num_envs

        if total_per_env < min_single_stance_frames:
            xor_progress = 0.0
        else:
            xor_progress = min(
                (total_per_env - min_single_stance_frames) / ramp_single_stance_frames,
                1.0,
            )

        # Fallback: common_step_counter-based schedule handles training resume
        # where the in-memory accumulator starts from zero.
        step_progress = min(
            env.common_step_counter / (safety_steps_per_env * num_envs), 1.0
        )

        base_progress = max(xor_progress, step_progress)

        # Velocity-tracking gate: scale down the weight when the population's
        # average tracking score is poor, so the weight only ramps up once
        # the robot can actually track commanded velocities.
        try:
            command = env.command_manager.get_term("base_velocity")
            tracking_score = command.metrics["tracking_exp_vel_xy"].mean().item()
            if tracking_score < vel_tracking_threshold or vel_tracking_target <= vel_tracking_threshold:
                vel_factor = 0.0 if tracking_score < vel_tracking_threshold else 1.0
            else:
                vel_factor = min(
                    (tracking_score - vel_tracking_threshold)
                    / (vel_tracking_target - vel_tracking_threshold),
                    1.0,
                )
        except (ValueError, KeyError):
            vel_factor = 1.0
            tracking_score = 0.0

        progress = base_progress * vel_factor
        new_weight = start_weight + (end_weight - start_weight) * progress
        self._term_cfg.weight = new_weight
        return {
            "weight": new_weight,
            "xor_progress": xor_progress,
            "step_progress": step_progress,
            "vel_factor": vel_factor,
            "base_progress": base_progress,
            "progress": progress,
            "tracking_score": tracking_score,
        }
