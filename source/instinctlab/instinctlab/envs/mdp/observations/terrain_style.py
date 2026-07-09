"""Observation exposing each env's AMP style index, derived from its subterrain.

In the Multi-AMP / multi-discriminator setup, every env is permanently bound to one terrain
group (e.g. flat vs. stairs). The training curriculum only changes an env's terrain *level*
(row), never its *type* (column), so the style is constant for the whole run and is computed
once then cached on the env.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from instinctlab.motion_reference.motion_files.terrain_aware_amass_motion import (
    _build_col_to_subterrain_name,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def amp_terrain_style(
    env: "ManagerBasedRLEnv",
    subterrain_to_style: dict[str, int],
    default_style: int = 0,
) -> torch.Tensor:
    """Per-env active AMP style index, shape ``(num_envs, 1)``.

    Args:
        subterrain_to_style: maps sub_terrain name (a key of
            ``scene.terrain.cfg.terrain_generator.sub_terrains``) to a style index. Subterrains
            not present here fall back to ``default_style``.
        default_style: style index used for unmapped subterrains, or when the terrain has no
            generator / no ``terrain_types`` (e.g. a plane).
    """
    cache = getattr(env, "_amp_terrain_style_cache", None)
    if cache is not None:
        return cache

    terrain = env.scene.terrain
    # kept as float so it flows through the ObservationManager like any other obs; the algorithm
    # casts it back to long. Style values are small integers, exactly representable as float.
    style = torch.full((env.num_envs, 1), float(default_style), dtype=torch.float, device=env.device)

    terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_gen_cfg is not None and terrain_types is not None:
        # The column -> sub_terrain mapping only holds for the deterministic curriculum layout;
        # with curriculum=False sub-terrains are placed randomly per tile and the mapping is invalid.
        if subterrain_to_style and not terrain_gen_cfg.curriculum:
            raise ValueError(
                "[amp_terrain_style] subterrain_to_style requires terrain_generator.curriculum=True "
                "(deterministic column layout); got curriculum=False."
            )
        terrain_types = terrain_types.to(env.device)
        col_to_name = _build_col_to_subterrain_name(terrain_gen_cfg)
        for col_idx, name in enumerate(col_to_name):
            if name in subterrain_to_style:
                style[terrain_types == col_idx] = float(subterrain_to_style[name])

        unique_types = torch.unique(terrain_types).tolist()
        counts = {col_to_name[int(t)]: int((terrain_types == t).sum().item()) for t in unique_types}
        print(
            f"[amp_terrain_style] style map={subterrain_to_style}, default={default_style}; "
            f"env counts by subterrain={counts}"
        )
    else:
        print(
            "[amp_terrain_style] terrain has no generator / terrain_types; "
            f"all envs use default style {default_style}."
        )

    env._amp_terrain_style_cache = style
    return style
