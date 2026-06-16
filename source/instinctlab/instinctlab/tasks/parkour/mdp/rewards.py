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

def foot_collision_penalty(
    env: ManagerBasedRLEnv,
    left_fwd_scanner_cfg: SceneEntityCfg,
    right_fwd_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    d_unsafe: float = 0.15,          # 惩罚生效距离阈值 (m)
) -> torch.Tensor:
    """
    计算脚与楼梯竖板碰撞的密集惩罚。
    返回形状 (num_envs,) 的张量，值为非正数（惩罚）。
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取左右脚的位置和速度（假设 body_ids[0]=左脚, body_ids[1]=右脚）
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[:2], :]       # (N, 2, 3)
    foot_vel = asset.data.body_vel_w[:, asset_cfg.body_ids[:2], :]       # (N, 2, 3)

    # 获取左右前向射线传感器
    left_scanner: RayCaster = env.scene[left_fwd_scanner_cfg.name]
    right_scanner: RayCaster = env.scene[right_fwd_scanner_cfg.name]

    # 射线击中点的位置 (N, num_rays, 3)  未击中则为 inf
    left_hits = left_scanner.data.ray_hits_w   # (N, num_rays, 3)
    right_hits = right_scanner.data.ray_hits_w
    # 将 inf 替换为 NaN 以便后续处理
    left_hits = torch.where(torch.isinf(left_hits), torch.nan, left_hits)
    right_hits = torch.where(torch.isinf(right_hits), torch.nan, right_hits)

    penalties = []
    for scanner_hits, foot_pos_xy, foot_vel_xy in zip(
        [left_hits, right_hits],
        foot_pos[:, :, :2].unbind(dim=1),  # 只取 XY, 按脚维度拆分 → 2个 (N, 2) 张量
        foot_vel[:, :, :2].unbind(dim=1)
    ):
        # foot_pos_xy: (N, 2), foot_vel_xy: (N, 2)
        v_norm = torch.norm(foot_vel_xy, dim=-1, keepdim=True)  # (N, 1)
        valid_foot = (v_norm > 0.01).squeeze(-1)

        # 对每个环境、每条射线，计算从脚到击中点的向量
        # scanner_hits: (N, R, 3) -> 取 XY，然后计算向量
        hit_xy = scanner_hits[..., :2]   # (N, R, 2)
        foot_xy_exp = foot_pos_xy.unsqueeze(1)  # (N, 1, 2)
        d_xy = hit_xy - foot_xy_exp       # (N, R, 2)
        dist = torch.norm(d_xy, dim=-1)   # (N, R)

        # 只考虑距离小于 d_unsafe 且击中有效的射线
        valid_hit = ~torch.isnan(hit_xy[..., 0]) & (dist < d_unsafe)

        # 计算每个射线的方向角与脚速度方向的夹角
        # 脚速度方向角
        foot_angle = torch.atan2(foot_vel_xy[..., 1], foot_vel_xy[..., 0])  # (N,)
        # 射线方向角
        ray_angle = torch.atan2(d_xy[..., 1], d_xy[..., 0])                 # (N, R)
        # 角度差，取绝对值，范围 [0, pi]
        angle_diff = torch.abs(ray_angle - foot_angle.unsqueeze(1))
        angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
        within_cone = angle_diff < (15.0 * torch.pi / 180.0)   # 30° 锥形半角15°

        # 综合有效条件: 击中且距离近且在锥形内
        effective = valid_hit & within_cone

        has_any = effective.any(dim=1)
        if not has_any.any():
            penalties.append(torch.zeros(env.num_envs, device=env.device))
            continue

        # 找出每个环境中距离最近的障碍物 (在有效射线中)
        # 将无效射线的距离设为无穷大
        dist_masked = torch.where(effective, dist, torch.inf)
        min_dist, min_idx = torch.min(dist_masked, dim=1)   # (N,)
        min_idx = torch.where(has_any, min_idx, 0)

        # 获取对应最近障碍物的 d_xy 和 击中点坐标
        # 使用 gather 或 索引
        batch_idx = torch.arange(env.num_envs, device=env.device)
        d_xy_min = d_xy[batch_idx, min_idx]  # (N, 2)
        # 计算基础惩罚 p_colli = max(0, v·d / |d|)
        v_dot_d = torch.sum(foot_vel_xy * d_xy_min, dim=-1)  # (N,)
        p_colli = torch.clamp(v_dot_d / (torch.norm(d_xy_min, dim=-1) + 1e-6), min=0.0)

        # 安全距离加权项 d_colli = max(0, 1 - |d|/d_unsafe)
        d_colli = torch.clamp(1 - min_dist / d_unsafe, min=0.0)

        # 最终惩罚 (负值)
        penalty = -p_colli * d_colli
        penalty = penalty * valid_foot.float()
        penalty = penalty * has_any.float()
        penalties.append(penalty)

    # 左右脚惩罚之和
    total_penalty = penalties[0] + penalties[1]
    return total_penalty


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)
