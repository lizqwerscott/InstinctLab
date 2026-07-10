from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


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


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, grace_time: float = 0.04) -> torch.Tensor:
    """Penalize sustained flight phases in which neither foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("ContactSensor must enable track_air_time for no_fly.")

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    simultaneous_air_time = torch.min(air_time, dim=1).values
    return (simultaneous_air_time > grace_time).float()


def feet_air_contact_time_variance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    vel_threshold: float = 0.15,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Penalize unequal completed air and contact durations between the feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise RuntimeError("ContactSensor must enable track_air_time for air/contact time variance.")

    last_air_time = torch.clamp(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], max=max_time)
    last_contact_time = torch.clamp(contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids], max=max_time)
    variance = torch.var(last_air_time, dim=1, unbiased=False) + torch.var(
        last_contact_time, dim=1, unbiased=False
    )

    command = env.command_manager.get_command(command_name)
    is_moving = torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return variance * is_moving


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ratio: float = 3.0,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet hitting vertical edges based on abnormal horizontal contact force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]

    lateral_force = torch.norm(forces[..., :2], dim=-1)
    vertical_force = torch.abs(forces[..., 2])
    total_force = torch.norm(forces, dim=-1)

    in_contact = total_force > contact_threshold
    stumble = in_contact & (lateral_force > ratio * vertical_force)
    return torch.sum(torch.any(stumble, dim=1).float(), dim=-1)


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
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
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
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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


class foothold_support_penalty(ManagerTermBase):
    """Penalize insufficient sole support once after each foot settles on a foothold."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        contact_sensor_cfg = params["contact_sensor_cfg"]
        asset_cfg = params["asset_cfg"]

        if len(contact_sensor_cfg.body_ids) != 2 or len(asset_cfg.body_ids) != 2:
            raise ValueError("foothold_support_penalty requires exactly two ordered foot bodies.")
        if not 0.0 < params["support_target"] <= 1.0:
            raise ValueError("support_target must be in (0, 1].")
        if params["epsilon_h"] < 0.0 or params["temperature"] <= 0.0:
            raise ValueError("epsilon_h must be non-negative and temperature must be positive.")
        if not 0.0 <= params["min_settle_time"] <= params["max_settle_time"]:
            raise ValueError("settle times must satisfy 0 <= min_settle_time <= max_settle_time.")

        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        if not contact_sensor.cfg.track_air_time:
            raise RuntimeError("ContactSensor must enable track_air_time for foothold support evaluation.")

        self._pending = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self._was_in_contact = torch.zeros_like(self._pending)
        self._sole_offset = torch.tensor(params["sole_offset"], dtype=torch.float, device=env.device)
        self._sole_normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = False
        self._was_in_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor_cfg: SceneEntityCfg,
        left_height_scanner_cfg: SceneEntityCfg,
        right_height_scanner_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        sole_offset: tuple[float, float, float],
        epsilon_h: float,
        temperature: float,
        support_target: float,
        min_settle_time: float,
        max_settle_time: float,
        max_vertical_speed: float,
        max_roll_pitch_rate: float,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        left_scanner = env.scene.sensors[left_height_scanner_cfg.name]
        right_scanner = env.scene.sensors[right_height_scanner_cfg.name]

        contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
        in_contact = contact_time > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, contact_sensor_cfg.body_ids]
        first_contact |= in_contact & ~self._was_in_contact
        first_contact &= env.episode_length_buf.unsqueeze(-1) > 1
        self._pending |= first_contact
        self._pending &= in_contact

        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        foot_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
        foot_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
        foot_ang_vel_b = quat_apply_inverse(
            foot_quat_w.reshape(-1, 4), foot_ang_vel_w.reshape(-1, 3)
        ).reshape_as(foot_ang_vel_w)
        is_stable = (torch.abs(foot_lin_vel_w[..., 2]) <= max_vertical_speed) & (
            torch.norm(foot_ang_vel_b[..., :2], dim=-1) <= max_roll_pitch_rate
        )
        ready = (contact_time >= min_settle_time) & is_stable
        timed_out = contact_time >= max_settle_time
        evaluate = self._pending & (ready | timed_out)

        foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
        sole_offset_w = quat_apply(
            foot_quat_w.reshape(-1, 4), self._sole_offset.expand(foot_quat_w.numel() // 4, -1)
        ).reshape_as(foot_pos_w)
        sole_center_w = foot_pos_w + sole_offset_w
        sole_normal_w = quat_apply(
            foot_quat_w.reshape(-1, 4), self._sole_normal.expand(foot_quat_w.numel() // 4, -1)
        ).reshape_as(foot_pos_w)

        ray_hits_w = torch.stack((left_scanner.data.ray_hits_w, right_scanner.data.ray_hits_w), dim=1)
        valid_hits = torch.isfinite(ray_hits_w).all(dim=-1)
        hit_x = torch.where(valid_hits, ray_hits_w[..., 0], sole_center_w[..., 0].unsqueeze(-1))
        hit_y = torch.where(valid_hits, ray_hits_w[..., 1], sole_center_w[..., 1].unsqueeze(-1))
        terrain_height = torch.where(valid_hits, ray_hits_w[..., 2], torch.zeros_like(ray_hits_w[..., 2]))

        normal_z = sole_normal_w[..., 2]
        safe_normal_z = torch.where(
            torch.abs(normal_z) > 1.0e-4, normal_z, torch.full_like(normal_z, 1.0e-4)
        )
        sole_height = sole_center_w[..., 2].unsqueeze(-1) - (
            sole_normal_w[..., 0].unsqueeze(-1) * (hit_x - sole_center_w[..., 0].unsqueeze(-1))
            + sole_normal_w[..., 1].unsqueeze(-1) * (hit_y - sole_center_w[..., 1].unsqueeze(-1))
        ) / safe_normal_z.unsqueeze(-1)
        height_error = torch.abs(sole_height - terrain_height)
        point_support = torch.sigmoid((epsilon_h - height_error) / temperature) * valid_hits
        support_score = torch.mean(point_support, dim=-1)
        support_deficit = torch.clamp((support_target - support_score) / support_target, min=0.0)
        penalty = torch.square(support_deficit) * evaluate

        self._pending[evaluate] = False
        self._was_in_contact[:] = in_contact
        return torch.sum(penalty, dim=-1) / env.step_dt


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)
