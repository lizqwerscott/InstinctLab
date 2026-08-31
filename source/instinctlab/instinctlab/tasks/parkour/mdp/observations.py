from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv


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
