from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gait_frequency(
    env: ManagerBasedEnv,
    action_name: str = "gait_frequency",
) -> torch.Tensor:
    """Return the EMA-filtered gait frequency produced by the action term."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.filtered_frequency


def gait_phase(
    env: ManagerBasedEnv,
    action_name: str = "gait_frequency",
) -> torch.Tensor:
    """Return sine/cosine features for the gait phase produced by the action term."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.phase_features


def gait_ratio(
    env: ManagerBasedEnv,
    action_name: str = "gait_frequency",
) -> torch.Tensor:
    """Return the EMA-filtered single-support ratio produced by the action term."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.filtered_ratio


def gait_ss_ds_sign(
    env: ManagerBasedEnv,
    action_name: str = "gait_frequency",
) -> torch.Tensor:
    """Return +1 in single support / -1 in double support (Egle-style phase sign)."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.ss_ds_sign
