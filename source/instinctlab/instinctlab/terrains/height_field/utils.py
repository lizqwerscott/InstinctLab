# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        mesh = meshes[0]
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = [mesh]

        # Get mesh bounds
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # Left wall
        if np.random.uniform() < cfg.wall_prob[0]:
            left_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(left_wall)

        # Right wall
        if np.random.uniform() < cfg.wall_prob[1]:
            right_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(right_wall)

        # Front wall
        if np.random.uniform() < cfg.wall_prob[2]:
            front_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(front_wall)

        # Back wall
        if np.random.uniform() < cfg.wall_prob[3]:
            back_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(back_wall)

        return result_meshes, origin

    return wrapper

def generate_stairs_side_wall(func: Callable) -> Callable:
    """Wrapper to add side walls strictly matching the length and width of the stairs.

    Fully supports tuple/list type configurations for domain randomization.
    """

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg):
        if cfg is None or not hasattr(cfg, "side_wall_prob"):
            return func(difficulty, cfg)

        if isinstance(cfg.per_step_length, (list, tuple)):
            per_step_length_m = cfg.per_step_length[0] + difficulty * (cfg.per_step_length[1] - cfg.per_step_length[0])
        else:
            per_step_length_m = cfg.per_step_length

        if isinstance(cfg.num_steps, (list, tuple)):
            num_steps_actual = cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0])
        else:
            num_steps_actual = cfg.num_steps
        num_steps_int = int(num_steps_actual)

        # 执行底层函数拿到裸楼梯矩阵
        hf_raw = func(difficulty, cfg)

        width_pixels = hf_raw.shape[0]   # X 轴总像素
        length_pixels = hf_raw.shape[1]  # Y 轴总像素

        # 使用解析出的确切浮点数，安全计算 X 轴起止切片
        if hasattr(cfg, "platform_length") and hasattr(cfg, "horizontal_scale"):
            platform_length_px = int(cfg.platform_length / cfg.horizontal_scale)
            per_step_length_px = int(per_step_length_m / cfg.horizontal_scale) # 转换为离散像素，此时绝对安全！

            # 对齐原函数的 num_steps 约束条件，防止越界
            num_steps_int = min(num_steps_int, (width_pixels - platform_length_px) // (2 * per_step_length_px))

            middle_x = width_pixels // 2
            start_x_up = middle_x - platform_length_px // 2
            start_x_down = start_x_up + platform_length_px

            # 精确锁定楼梯在 X 轴上的像素边界
            stair_start_x = max(0, start_x_up - num_steps_int * per_step_length_px)
            stair_end_x = min(width_pixels, start_x_down + num_steps_int * per_step_length_px)

            x_slice = slice(stair_start_x, stair_end_x)
        else:
            x_slice = slice(None)

        # 计算 Y 轴（左右方向）楼梯宽度边界
        per_step_width_m = getattr(cfg, "per_step_width", None)
        if per_step_width_m is None:
            per_step_width_m = cfg.size[1]

        per_step_width = int(per_step_width_m / cfg.horizontal_scale)
        middle_y = length_pixels // 2
        start_y = middle_y - per_step_width // 2
        end_y = start_y + per_step_width

        thickness_pixels = max(1, int(cfg.side_wall_thickness / cfg.horizontal_scale))
        max_stair_height = np.max(hf_raw)
        wall_height_discrete = max_stair_height + int(cfg.side_wall_height / cfg.vertical_scale)

        if np.random.uniform() < cfg.side_wall_prob[0]:
            y_left_start = max(0, start_y - thickness_pixels)
            hf_raw[x_slice, y_left_start:start_y] = wall_height_discrete

        if np.random.uniform() < cfg.side_wall_prob[1]:
            y_right_end = min(length_pixels, end_y + thickness_pixels)
            hf_raw[x_slice, end_y:y_right_end] = wall_height_discrete

        return hf_raw

    return wrapper
