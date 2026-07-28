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

    A ``common_step_counter``-based fallback (``safety_step_counter``)
    prevents the weight from remaining at zero after a training resume,
    where the in-memory accumulator would start from scratch.

    The default ``end_weight=None`` picks up whatever weight is set in the
    config, so the curriculum stays in sync if the config changes.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        try:
            reward_term_name = cfg.params.get("reward_term_name", "foothold_proximity")
            self._term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
            self._initial_weight = self._term_cfg.weight
            self._has_term = True
        except ValueError:
            self._has_term = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_weight: float = 0.0,
        end_weight: float | None = None,
        min_single_stance_frames: int = 1_000_000,
        ramp_single_stance_frames: int = 10_000_000,
        safety_step_counter: int = 200_000_000,
    ) -> float:
        if not self._has_term:
            return start_weight

        if end_weight is None:
            end_weight = self._initial_weight

        total = self._term_cfg.func.cumulative_single_stance_frames
        if total < min_single_stance_frames:
            xor_progress = 0.0
        else:
            xor_progress = min(
                (total - min_single_stance_frames) / ramp_single_stance_frames, 1.0
            )

        # Fallback: common_step_counter-based schedule handles training resume
        # where the in-memory accumulator starts from zero.
        step_progress = min(env.common_step_counter / safety_step_counter, 1.0)

        progress = max(xor_progress, step_progress)
        new_weight = start_weight + (end_weight - start_weight) * progress
        self._term_cfg.weight = new_weight
        return new_weight
