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
    """Two-factor gait curriculum for the combined foothold reward.

    Two independent ramps, matching the observed training timeline
    (stand 0~2k → hop 2k~8k → alternating gait 8k+):

    - ``movement_factor``: population mean of (achieved horizontal speed /
      commanded speed), EMA-free instantaneous mean. It ramps the external
      ``RewTerm.weight`` AND the gait-discipline internal weights
      (``anti_hop_weight``, ``wrong_foot_weight``, ``com_bounce_weight``,
      ``swing_onset_weight``). Because hopping satisfies velocity tracking,
      gating discipline behind tracking quality would only turn on the
      anti-hop guidance once hopping is already entrenched (~8k). Keying it
      on *movement* instead activates the guidance as soon as the robot
      starts moving (~2-4k) — hopping is caught while it is still young
      (preventive), and the penalty strength scales with how much the robot
      actually moves, so a robot that cannot walk yet is not penalized
      (no deadlock).

    - ``vel_factor``: velocity-tracking EMA (as before). It ramps the
      precision internal weights (``proximity_weight``, ``bezier_weight``)
      so precise foothold placement is only demanded once the robot tracks
      velocity well (corrective, ~8k+).

    The curriculum mutates ``term_cfg.params`` directly; the reward manager
    re-reads them on every frame (``term_cfg.func(env, **term_cfg.params)``),
    so the changes take effect immediately.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_term_name = cfg.params.get("reward_term_name", "foothold")
        self._command_name = cfg.params.get("command_name", "base_velocity")
        self._term_cfg = None
        self._initial_weight = None
        self._vel_tracking_ema = 0.0
        self._base_internal: dict[str, float] = {}

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
        movement_start: float = 0.15,
        movement_end: float = 0.5,
    ) -> dict:
        if self._term_cfg is None:
            try:
                self._term_cfg = env.reward_manager.get_term_cfg(self._reward_term_name)
                self._initial_weight = self._term_cfg.weight
                p = self._term_cfg.params
                # 快照配置里的基础内部权重, 课程在此基础上乘 ramp 因子
                self._base_internal = {
                    "anti_hop_weight": p.get("anti_hop_weight", 1.0),
                    "wrong_foot_weight": p.get("wrong_foot_weight", 1.0),
                    "com_bounce_weight": p.get("com_bounce_weight", 0.5),
                    "swing_onset_weight": p.get("swing_onset_weight", 0.3),
                    "proximity_weight": p.get("proximity_weight", 1.0),
                    "bezier_weight": p.get("bezier_weight", 1.0),
                }
            except ValueError:
                pass

        if self._term_cfg is None or not self._base_internal:
            return {
                "weight": start_weight,
                "vel_factor": 1.0,
                "vel_tracking_ema": self._vel_tracking_ema,
                "instant_tracking": 0.0,
                "movement": 0.0,
                "movement_factor": 0.0,
            }

        if end_weight is None:
            end_weight = self._initial_weight

        try:
            command = env.command_manager.get_term(self._command_name)
            robot = command.robot
        except (ValueError, KeyError, AttributeError):
            command = None
            robot = None

        # ---- Factor 2: velocity-tracking EMA (precision ramp) ------------
        instant_tracking = 0.0
        if command is not None and robot is not None:
            try:
                lin_vel_error = torch.sum(
                    torch.square(command.vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2]),
                    dim=1,
                )
                instant_tracking = torch.exp(-lin_vel_error / (command.cfg.lin_vel_metrics_std**2)).mean().item()
            except (ValueError, KeyError, AttributeError):
                instant_tracking = 0.0

        self._vel_tracking_ema = vel_ema_alpha * instant_tracking + (1.0 - vel_ema_alpha) * self._vel_tracking_ema

        if self._vel_tracking_ema < vel_tracking_threshold or vel_tracking_target <= vel_tracking_threshold:
            vel_factor = 0.0 if self._vel_tracking_ema < vel_tracking_threshold else 1.0
        else:
            vel_factor = min(
                (self._vel_tracking_ema - vel_tracking_threshold) / (vel_tracking_target - vel_tracking_threshold),
                1.0,
            )

        # ---- Factor 1: population movement level (discipline ramp) -------
        # 实际水平速度 / 指令速度 的群体均值 → 机器人一开始移动引导就生效。
        movement = 0.0
        if command is not None and robot is not None:
            try:
                v_robot = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
                v_cmd_norm = torch.norm(command.vel_command_b[:, :2], dim=1)
                ratio = (v_robot / (v_cmd_norm + 1e-3)).clamp(0.0, 1.0)
                movement = ratio.mean().item()
            except (ValueError, KeyError, AttributeError):
                movement = 0.0

        if movement < movement_start or movement_end <= movement_start:
            movement_factor = 0.0 if movement < movement_start else 1.0
        else:
            movement_factor = min((movement - movement_start) / (movement_end - movement_start), 1.0)

        # ---- Apply: external weight on movement, internal weights split ---
        new_weight = start_weight + (end_weight - start_weight) * movement_factor
        self._term_cfg.weight = new_weight

        base_w = self._base_internal
        # 纪律族: 随移动 ramp (预防性, 蹦跳刚出现就压制)
        discipline_family = ("anti_hop_weight", "wrong_foot_weight", "com_bounce_weight", "swing_onset_weight")
        for key in discipline_family:
            self._term_cfg.params[key] = base_w[key] * movement_factor
        # 精度族: 随速度跟踪 ramp (纠错性, 跟踪好了才要求精确落点)
        for key in ("proximity_weight", "bezier_weight"):
            self._term_cfg.params[key] = base_w[key] * vel_factor

        return {
            "weight": new_weight,
            "vel_factor": vel_factor,
            "vel_tracking_ema": self._vel_tracking_ema,
            "instant_tracking": instant_tracking,
            "movement": movement,
            "movement_factor": movement_factor,
        }
