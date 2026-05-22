from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

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
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_contact_flatness(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_clip: float = 0.08,
    flatness_threshold: float = 0.03,
) -> torch.Tensor:
    """Penalize a stance foot landing on uneven ground (stair edges / corners).

    For each foot in contact, a small grid of rays under the foot footprint samples the
    local terrain height. The std of those heights is ~0 on a flat patch and large when
    the foot straddles a stair edge (rays split between two step levels). Heights are
    clipped around the patch median so a single tall step (or a missed ray) cannot
    dominate the signal. A deadband ignores ordinary roughness below flatness_threshold.
    Calibration-free: it measures terrain, not foot/shoe geometry.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0

    reward = torch.zeros(env.num_envs, device=env.device)
    for i, scanner_cfg in enumerate((left_height_scanner_cfg, right_height_scanner_cfg)):
        scanner = env.scene[scanner_cfg.name]
        hits_z = scanner.data.ray_hits_w[..., 2]  # (N, P)
        # missed rays (inf) -> treat as foot level so they add no spurious penalty
        foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[i], 2:3]
        hits_z = torch.where(torch.isinf(hits_z), foot_z, hits_z)
        # clip around the patch median so a tall lower step cannot dominate the std
        center = torch.median(hits_z, dim=-1, keepdim=True).values
        hits_z = torch.clamp(hits_z, center - height_clip, center + height_clip)
        flatness = torch.std(hits_z, dim=-1)
        reward = reward + torch.relu(flatness - flatness_threshold) * is_contact[:, i]
    return reward


def _feet_pivot_overhang(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    left_pivot_scanner_cfg: SceneEntityCfg,
    right_pivot_scanner_cfg: SceneEntityCfg,
    support_tol: float = 0.03,
    overhang_clip: float = 0.1,
    settled_time: float = 0.08,
    support_topk: int = 3,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return per-foot normalized ankle-pivot overhang in [0, 1].

    The foot patch estimates the highest nearby support surface, while the pivot scanner
    checks the terrain directly under the ankle roll axis. Overhang is positive only when
    the support surface is above the pivot terrain by more than support_tol, which catches
    heel/toe edge support without penalizing normal ankle motion on a solid step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    )
    if settled_time > 0.0:
        contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
        is_contact = is_contact & (contact_time > settled_time)

    overhang = torch.zeros((env.num_envs, 2), device=env.device)
    scanner_pairs = (
        (left_height_scanner_cfg, left_pivot_scanner_cfg),
        (right_height_scanner_cfg, right_pivot_scanner_cfg),
    )
    for i, (height_scanner_cfg, pivot_scanner_cfg) in enumerate(scanner_pairs):
        support_hits_z = env.scene[height_scanner_cfg.name].data.ray_hits_w[..., 2]
        low_sentinel = support_hits_z.new_full((), -1.0e6)
        support_hits_z = torch.where(torch.isfinite(support_hits_z), support_hits_z, low_sentinel)
        topk = min(support_topk, support_hits_z.shape[-1])
        z_support = torch.topk(support_hits_z, k=topk, dim=-1).values.mean(dim=-1)

        pivot_hits_z = env.scene[pivot_scanner_cfg.name].data.ray_hits_w[..., 2]
        missed_pivot_z = (z_support - overhang_clip - support_tol).unsqueeze(-1)
        pivot_hits_z = torch.where(torch.isfinite(pivot_hits_z), pivot_hits_z, missed_pivot_z)
        z_pivot = torch.mean(pivot_hits_z, dim=-1)

        raw_overhang = torch.relu(z_support - z_pivot - support_tol)
        overhang[:, i] = torch.clamp(raw_overhang, max=overhang_clip) / overhang_clip

    return overhang * is_contact.float()


def feet_pivot_overhang(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    left_pivot_scanner_cfg: SceneEntityCfg,
    right_pivot_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_tol: float = 0.03,
    overhang_clip: float = 0.1,
    settled_time: float = 0.08,
    support_topk: int = 3,
) -> torch.Tensor:
    """Penalize stance feet whose ankle-roll pivot is unsupported by the current step."""
    del asset_cfg  # kept for config symmetry with feet_edge_contact_rotate
    overhang = _feet_pivot_overhang(
        env=env,
        contact_sensor_cfg=contact_sensor_cfg,
        left_height_scanner_cfg=left_height_scanner_cfg,
        right_height_scanner_cfg=right_height_scanner_cfg,
        left_pivot_scanner_cfg=left_pivot_scanner_cfg,
        right_pivot_scanner_cfg=right_pivot_scanner_cfg,
        support_tol=support_tol,
        overhang_clip=overhang_clip,
        settled_time=settled_time,
        support_topk=support_topk,
    )
    return torch.sum(torch.square(overhang), dim=-1)


def feet_edge_contact_rotate(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    left_pivot_scanner_cfg: SceneEntityCfg,
    right_pivot_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_tol: float = 0.03,
    overhang_clip: float = 0.1,
    settled_time: float = 0.08,
    support_topk: int = 3,
) -> torch.Tensor:
    """Penalize foot roll/pitch rotation only when the ankle pivot is overhanging an edge."""
    asset = env.scene[asset_cfg.name]
    overhang = _feet_pivot_overhang(
        env=env,
        contact_sensor_cfg=contact_sensor_cfg,
        left_height_scanner_cfg=left_height_scanner_cfg,
        right_height_scanner_cfg=right_height_scanner_cfg,
        left_pivot_scanner_cfg=left_pivot_scanner_cfg,
        right_pivot_scanner_cfg=right_pivot_scanner_cfg,
        support_tol=support_tol,
        overhang_clip=overhang_clip,
        settled_time=settled_time,
        support_topk=support_topk,
    )
    foot_ang_vel_xy = torch.norm(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(overhang * foot_ang_vel_xy, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)
