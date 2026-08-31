from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return (
        (dof_error - offset)
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
    )


def feet_close_xy_gauss(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1
    )
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(
            left_height.unsqueeze(-1) - left_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(
            right_height.unsqueeze(-1) - right_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


# ===========================================================================
# HugWBC gait phase clock and contact-swing rewards
#
# Reference: "A Unified and General Humanoid Whole-Body Controller for
# Versatile Locomotion" (arXiv:2503.04393), HugWBC open-source implementation
# (legged_gym/envs/h1/h1.py).
#
# The gait frequency, phase offset, stance fraction (duty cycle) and foot
# swing height are per-environment behavior commands sampled by
# PoseVelocityCommand (behavior_command, see commands/pose_velocity_command.py),
# exactly like HugWBC's extended behavior commands. Standing environments are
# the command's standing task mode (is_standing_env, sampled with probability
# rel_standing_envs at each command resample; velocity zeroed) — the same
# mechanism HugWBC uses for its 10% standing mode.
# ===========================================================================


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """CDF of the standard normal distribution N(0, 1)."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _polynomial_planer(t0, t1, x0, x1, v0=0, v1=0, a0=0, a1=0):
    """Quintic polynomial coefficients with zero velocity/acceleration at both ends.

    Mirrors ``_polynomial_planer`` from the HugWBC open-source implementation
    (legged_gym/envs/h1/h1.py). Used for the swing-height target (Eq. 10).
    """
    T = t1 - t0
    h = x1 - x0
    k0 = x0
    k1 = v0
    k2 = 0.5 * a0
    k3 = (20 * h - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * (T**2)) / (2 * (T**3))
    k4 = (-30 * h + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * (T**2)) / (2 * (T**4))
    k5 = (12 * h - 6 * (v1 + v0) * T + (a1 - a0) * (T**2)) / (2 * (T**5))
    return [k0, k1, k2, k3, k4, k5]


# ---------------------------------------------------------------------------
# Shared gait phase tracker
# ---------------------------------------------------------------------------

_TRACKER_ATTR = "_hugwbc_gait_tracker"


class GaitPhaseTracker:
    """Open-loop gait phase clock shared by all HugWBC-style terms.

    Faithful to HugWBC ``_step_contact_targets`` (h1.py):

    * the internal clock advances for **all** environments every step
      (``remainder(gait_indices + dt * frequencies, 1.0)``), so it idles
      through standing; the gait frequency comes from the per-environment
      behavior command (``behavior_command[:, 0]``);
    * standing environments (command ``is_standing_env``) have **both** feet'
      output phase zeroed (HugWBC zeroes ``foot_indices`` of standing envs);
    * the homogenized phase (Eq. 6) and the expected contact probability
      (Eq. 5) are computed with ``remainder(phi, 1)`` applied first, matching
      the source structure.

    The clock advances at most once per environment step (guarded by
    ``env.common_step_counter``) so observation and reward terms all read a
    consistent phase. Environments that just (re)started restart at 0.
    """

    def __init__(self, num_envs: int, device: torch.device, phase_sigma: float = 0.05):
        self._phase_sigma = float(phase_sigma)
        self._phase = torch.zeros(num_envs, device=device)
        self._is_standing = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._duty = torch.full((num_envs,), 0.5, device=device)
        self._offset = torch.full((num_envs,), 0.5, device=device)
        self._swing_height = torch.full((num_envs,), 0.08, device=device)
        self._last_step = -1

    # -- public API ------------------------------------------------------

    def update(self, env: "ManagerBasedRLEnv") -> None:
        """Advance the clock at most once per environment step."""
        if env.common_step_counter == self._last_step:
            return
        self._last_step = env.common_step_counter

        just_reset = env.episode_length_buf == 0
        command = env.command_manager.get_term("base_velocity")
        self._is_standing = command.is_standing_env
        behavior = command.behavior_command  # (N, 7): [freq, swing_h, h, pitch, yaw, offset, duty]
        self._offset = behavior[:, 5]
        self._duty = behavior[:, 6].clamp(1e-3, 1.0 - 1e-3)
        self._swing_height = behavior[:, 1]

        # Advance for ALL envs (idles through standing); reset restarts at 0.
        self._phase = torch.where(
            just_reset,
            torch.zeros_like(self._phase),
            torch.remainder(self._phase + behavior[:, 0] * env.step_dt, 1.0),
        )

    @property
    def raw_phase(self) -> torch.Tensor:
        """Raw phase (N, 2): [left, right].

        Walking envs: [phi, remainder(phi + phase_offset, 1)]. Standing envs:
        both feet held at 0, mirroring HugWBC's zeroing of the ``foot_indices``
        of standing environments.
        """
        both = torch.stack([self._phase, torch.remainder(self._phase + self._offset, 1.0)], dim=-1)
        return torch.where(self._is_standing.unsqueeze(-1), torch.zeros_like(both), both)

    def homogenized_phase(self) -> torch.Tensor:
        """Homogenized phase phi_bar (N, 2) (HugWBC Eq. 6)."""
        # Faithful structure: remainder first (no-op since phase is in [0, 1)).
        ph = torch.remainder(self.raw_phase, 1.0)
        duty = self._duty.unsqueeze(-1)
        return torch.where(
            ph < duty,
            ph * (0.5 / duty),
            0.5 + (ph - duty) * (0.5 / (1.0 - duty)),
        )

    def contact_prob(self) -> torch.Tensor:
        """Expected contact probability C(phi) (N, 2) (HugWBC Eq. 5)."""
        # Faithful structure: remainder before the CDF (no-op for phi_bar in [0, 1)).
        phi_bar = torch.remainder(self.homogenized_phase(), 1.0)
        sigma = self._phase_sigma
        Phi = _normal_cdf
        return Phi(phi_bar / sigma) * (1.0 - Phi((phi_bar - 0.5) / sigma)) + Phi((phi_bar - 1.0) / sigma) * (
            1.0 - Phi((phi_bar - 1.5) / sigma)
        )

    def clock_inputs(self) -> torch.Tensor:
        """Clock functions sin(2*pi*phi_bar) (N, 2), one per foot."""
        return torch.sin(2.0 * math.pi * self.homogenized_phase())

    @property
    def swing_height(self) -> torch.Tensor:
        """Per-environment foot swing height command (N,)."""
        return self._swing_height


def hugwbc_base_height_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Return squared local-terrain base-height tracking error."""
    robot = env.scene["robot"]
    scanner = env.scene[sensor_cfg.name]
    ray_heights = scanner.data.ray_hits_w[..., 2]
    ray_heights = torch.where(torch.isfinite(ray_heights), ray_heights, torch.zeros_like(ray_heights))
    terrain_height = ray_heights.mean(dim=-1)
    command = env.command_manager.get_command(command_name)
    desired_height = target_height + command[:, 5]
    error = robot.data.root_pos_w[:, 2] - terrain_height - desired_height
    reward = torch.square(error)
    standing = env.command_manager.get_term(command_name).is_standing_env
    return torch.where(standing, 3.0 * reward, reward)


def hugwbc_body_pitch_tracking(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Return squared body-pitch tracking error."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    pitch_axis = torch.zeros_like(robot.data.root_pos_w)
    pitch_axis[:, 1] = 1.0
    pitch_quat = quat_from_angle_axis(command[:, 6], pitch_axis)
    desired_projected_gravity = quat_apply_inverse(pitch_quat, robot.data.GRAVITY_VEC_W)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, robot.data.GRAVITY_VEC_W)
    return torch.sum(
        torch.square(projected_gravity[:, :2] - desired_projected_gravity[:, :2]),
        dim=-1,
    )


def hugwbc_waist_yaw_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return squared waist-yaw tracking error relative to the nominal pose."""
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_error = joint_error - command[:, 7:8]
    return torch.sum(torch.square(joint_error), dim=-1)


def hugwbc_feet_slip(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Return the HugWBC bounded stance foot-slip penalty."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        torch.max(
            torch.linalg.norm(
                contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids],
                dim=-1,
            ),
            dim=1,
        )[0]
        > threshold
    )
    robot = env.scene[asset_cfg.name]
    foot_velocity = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip_speed = torch.linalg.norm(foot_velocity, dim=-1) * contact
    return 1.0 - torch.exp(-torch.sum(slip_speed, dim=-1))


def hugwbc_feet_symmetry(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    phase_tolerance: float = 1.0e-5,
) -> torch.Tensor:
    """Return the phase-gated squared x-z foot symmetry error."""
    tracker = get_gait_tracker(env)
    tracker.update(env)
    phase = tracker.homogenized_phase()
    same_phase = torch.isclose(phase[:, 0], phase[:, 1], atol=phase_tolerance, rtol=0.0)
    robot = env.scene[asset_cfg.name]
    relative_positions = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w.unsqueeze(1)
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, relative_positions.shape[1], -1)
    positions_b = quat_apply_inverse(root_quat_w, relative_positions)
    difference = positions_b[:, 0][:, (0, 2)] - positions_b[:, 1][:, (0, 2)]
    return torch.sum(torch.square(difference), dim=-1) * same_phase


class HugWBCActionSmoothness(ManagerTermBase):
    """Return the squared second difference of policy actions."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        previous_action = env.action_manager.prev_action.clone()
        self._previous_action = previous_action
        self._previous_previous_action = previous_action.clone()

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        action = env.action_manager.action
        action_difference = action - 2.0 * self._previous_action + self._previous_previous_action
        self._previous_previous_action[:] = self._previous_action
        self._previous_action[:] = action
        return torch.sum(torch.square(action_difference), dim=-1)

    def reset(self, env_ids: Sequence[int] | slice):
        previous_action = self._env.action_manager.prev_action[env_ids]
        self._previous_action[env_ids] = previous_action
        self._previous_previous_action[env_ids] = previous_action


def get_gait_tracker(env: "ManagerBasedRLEnv", phase_sigma: float = 0.05) -> GaitPhaseTracker:
    """Return the env's shared gait tracker, creating it on first use.

    The first caller supplies ``phase_sigma``; later callers reuse the existing
    tracker (they must configure the same ``phase_sigma``).
    """
    tracker = getattr(env, _TRACKER_ATTR, None)
    if tracker is None:
        tracker = GaitPhaseTracker(num_envs=env.num_envs, device=env.device, phase_sigma=phase_sigma)
        setattr(env, _TRACKER_ATTR, tracker)
    return tracker


# ---------------------------------------------------------------------------
# Observation term: gait clock functions
# ---------------------------------------------------------------------------


class GaitPhaseClockTerm(ManagerTermBase):
    """Observation term for the gait clock functions ``sin(2*pi*phi_bar)``.

    Output shape (N, 2) — [left foot, right foot] contact indicators used by
    HugWBC to let the policy observe the gait cycle phase.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._tracker = get_gait_tracker(env, phase_sigma=cfg.params.get("phase_sigma", 0.05))

    def __call__(self, env: "ManagerBasedRLEnv", phase_sigma: float = 0.05) -> torch.Tensor:
        self._tracker.update(env)
        return self._tracker.clock_inputs()


# ---------------------------------------------------------------------------
# Reward term: periodic contact-swing reward (Eq. 8)
# ---------------------------------------------------------------------------


class HugWBCContactSwingReward(ManagerTermBase):
    """Periodic contact-swing reward (HugWBC Eq. 8).

    ``component="force"``:  -sum_i (1 - C(phi_i)) (1 - exp(-f_i^2 / force_sigma)) / 2
    ``component="vel"``:    -sum_i C(phi_i) (1 - exp(-||v_i||^2 / vel_sigma)) / 2

    The external ``RewTerm`` weight applies per component (2.0 / 4.0 in the
    HugWBC configuration). ``vel_use_xy`` follows the paper's Eq. (8), which
    penalizes the horizontal foot velocity during stance.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._sensor_cfg = cfg.params["sensor_cfg"]
        self._component = cfg.params.get("component", "force")
        self._force_sigma = cfg.params.get("force_sigma", 50.0)
        self._vel_sigma = cfg.params.get("vel_sigma", 5.0)
        self._vel_use_xy = cfg.params.get("vel_use_xy", True)
        self._tracker = get_gait_tracker(env, phase_sigma=cfg.params.get("phase_sigma", 0.05))

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        phase_sigma: float = 0.05,
        asset_cfg: SceneEntityCfg | None = None,
        sensor_cfg: SceneEntityCfg | None = None,
        component: str = "force",
        force_sigma: float = 50.0,
        vel_sigma: float = 5.0,
        vel_use_xy: bool = True,
    ) -> torch.Tensor:
        self._tracker.update(env)
        if self._component == "force":
            contact: ContactSensor = env.scene.sensors[self._sensor_cfg.name]
            forces = contact.data.net_forces_w_history[:, -1, self._sensor_cfg.body_ids]  # (N, 2, 3)
            f_norm = torch.norm(forces, dim=-1)
            c = self._tracker.contact_prob()
            reward = -((1.0 - c) * (1.0 - torch.exp(-(f_norm**2) / self._force_sigma))).sum(dim=-1) / 2.0
        else:
            asset = env.scene[self._asset_cfg.name]
            vel = asset.data.body_lin_vel_w[:, self._asset_cfg.body_ids]  # (N, 2, 3)
            if self._vel_use_xy:
                vel = vel[..., :2]
            v_norm = torch.norm(vel, dim=-1)
            c = self._tracker.contact_prob()
            reward = -(c * (1.0 - torch.exp(-(v_norm**2) / self._vel_sigma))).sum(dim=-1) / 2.0
        return reward


# ---------------------------------------------------------------------------
# Reward term: foot swing-height trajectory (Eq. 9/10)
# ---------------------------------------------------------------------------


class HugWBCFeetClearanceReward(ManagerTermBase):
    """Foot swing-height trajectory reward (HugWBC Eq. 9/10).

    The target swing height follows a quintic polynomial over the homogenized
    phase inside the swing half-cycle (apex at phi_bar = 0.75), masked by the
    swing weight ``(1 - C(phi))``. The swing height is the per-environment
    behavior command (HugWBC reads it from ``commands[:, 6]``; here from
    ``behavior_command[:, 1]``). Foot height is measured relative to the
    ground height sampled by the left/right height scanners.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._left_scanner_cfg = cfg.params["left_height_scanner_cfg"]
        self._right_scanner_cfg = cfg.params["right_height_scanner_cfg"]
        self._base_height = cfg.params.get("base_height", 0.07)
        self._clip_max = cfg.params.get("clip_max", 0.1)
        self._tracker = get_gait_tracker(env, phase_sigma=cfg.params.get("phase_sigma", 0.05))

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        phase_sigma: float = 0.05,
        asset_cfg: SceneEntityCfg | None = None,
        left_height_scanner_cfg: SceneEntityCfg | None = None,
        right_height_scanner_cfg: SceneEntityCfg | None = None,
        base_height: float = 0.07,
        clip_max: float = 0.1,
    ) -> torch.Tensor:
        self._tracker.update(env)
        phi_bar = self._tracker.homogenized_phase()  # (N, 2)

        # triangular swing window around the apex phi_bar = 0.75
        phases = torch.clamp(0.75 - torch.abs(phi_bar - 0.75), 0.0, 1.0)  # (N, 2)
        coef = _polynomial_planer(0.5, 0.75, 0, 1)
        p = phases - 0.5
        curve = coef[0] + coef[1] * p + coef[2] * p**2 + coef[3] * p**3 + coef[4] * p**4 + coef[5] * p**5
        curve = torch.where(phases < 0.5, torch.zeros_like(curve), curve)
        target_height = self._tracker.swing_height.unsqueeze(-1) * curve + self._base_height  # (N, 2)

        foot_height = self._foot_height_above_ground(env)  # (N, 2)
        c = self._tracker.contact_prob()
        reward = torch.sum((target_height - foot_height) ** 2 * (1.0 - c), dim=-1)
        return reward.clamp(max=self._clip_max)

    def _foot_height_above_ground(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        """(N, 2) foot height above the ground under each foot."""
        asset = env.scene[self._asset_cfg.name]
        left = env.scene[self._left_scanner_cfg.name]
        right = env.scene[self._right_scanner_cfg.name]
        left_gz = torch.where(torch.isinf(left.data.ray_hits_w[..., 2]), 0.0, left.data.ray_hits_w[..., 2]).mean(dim=-1)
        right_gz = torch.where(
            torch.isinf(right.data.ray_hits_w[..., 2]),
            0.0,
            right.data.ray_hits_w[..., 2],
        ).mean(dim=-1)
        left_z = asset.data.body_pos_w[:, self._asset_cfg.body_ids[0], 2]
        right_z = asset.data.body_pos_w[:, self._asset_cfg.body_ids[1], 2]
        return torch.stack([left_z - left_gz, right_z - right_gz], dim=-1)
