from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def command_slice(
    env: ManagerBasedEnv,
    command_name: str,
    start: int,
    end: int,
) -> torch.Tensor:
    """Return a slice of the Parkour task and behavior command."""
    command = env.command_manager.get_command(command_name)
    if command.ndim != 2:
        raise ValueError(f"Expected a batched command tensor, got shape {tuple(command.shape)}")
    if not 0 <= start < end <= command.shape[1]:
        raise ValueError(f"Invalid command slice [{start}:{end}] for command shape {tuple(command.shape)}")
    return command[:, start:end]


def base_height_error(
    env: ManagerBasedEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Return the simulated base-height error relative to the local terrain."""
    robot = env.scene["robot"]
    scanner = env.scene[sensor_cfg.name]
    ray_heights = scanner.data.ray_hits_w[..., 2]
    ray_heights = torch.where(torch.isfinite(ray_heights), ray_heights, torch.zeros_like(ray_heights))
    terrain_height = ray_heights.mean(dim=-1, keepdim=True)
    body_height_offset = env.command_manager.get_command(command_name)[:, 5:6]
    error = robot.data.root_pos_w[:, 2:3] - terrain_height - target_height - body_height_offset
    return error.clamp(-0.5, 0.5)


def foot_clearance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return left and right foot heights above their local terrain."""
    robot = env.scene[asset_cfg.name]
    left_scanner = env.scene[left_height_scanner_cfg.name]
    right_scanner = env.scene[right_height_scanner_cfg.name]

    left_ground = torch.where(
        torch.isfinite(left_scanner.data.ray_hits_w[..., 2]),
        left_scanner.data.ray_hits_w[..., 2],
        torch.zeros_like(left_scanner.data.ray_hits_w[..., 2]),
    ).mean(dim=-1)
    right_ground = torch.where(
        torch.isfinite(right_scanner.data.ray_hits_w[..., 2]),
        right_scanner.data.ray_hits_w[..., 2],
        torch.zeros_like(right_scanner.data.ray_hits_w[..., 2]),
    ).mean(dim=-1)
    foot_heights = torch.stack(
        [
            robot.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - left_ground,
            robot.data.body_pos_w[:, asset_cfg.body_ids[1], 2] - right_ground,
        ],
        dim=-1,
    )
    return foot_heights.clamp(-1.0, 1.0)


def friction_coefficients(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the mean static friction coefficient of the randomized asset materials."""
    asset = env.scene[asset_cfg.name]
    material_properties = asset.root_physx_view.get_material_properties()
    return material_properties[..., 0].mean(dim=-1, keepdim=True)


def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the latest three-axis contact force for each selected foot."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    return forces.flatten(start_dim=1)


def collision_states(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Return binary collision states for selected non-foot bodies."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(forces, dim=-1) > threshold).to(dtype=forces.dtype)
