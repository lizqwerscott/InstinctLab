from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import GaitFrequencyActionCfg


class GaitFrequencyAction(ActionTerm):
    """Map a normalized policy action to a filtered gait frequency and phase."""

    cfg: GaitFrequencyActionCfg

    def __init__(self, cfg: GaitFrequencyActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        frequency_min, frequency_max = cfg.frequency_range
        if frequency_min < 0.0 or frequency_max <= frequency_min:
            raise ValueError(f"frequency_range must satisfy 0 <= min < max, got {cfg.frequency_range}")
        if not 0.0 < cfg.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {cfg.ema_alpha}")
        if not frequency_min <= cfg.initial_frequency <= frequency_max:
            raise ValueError(
                f"initial_frequency must be within frequency_range, got {cfg.initial_frequency} and"
                f" {cfg.frequency_range}"
            )

        self._frequency_min = frequency_min
        self._frequency_max = frequency_max
        self._ema_alpha = cfg.ema_alpha
        self._control_dt = env.step_dt
        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._filtered_frequency = torch.full(
            (env.num_envs, 1),
            cfg.initial_frequency,
            device=self.device,
            dtype=torch.float32,
        )
        self._phase = torch.full(
            (env.num_envs, 1),
            cfg.initial_phase % 1.0,
            device=self.device,
            dtype=torch.float32,
        )
        self._processed_actions[:] = self._filtered_frequency

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw policy actions before frequency processing."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The EMA-filtered frequency actions exposed to the action manager."""
        return self._processed_actions

    @property
    def filtered_frequency(self) -> torch.Tensor:
        """The EMA-filtered frequency in Hz, with shape (num_envs, 1)."""
        return self._filtered_frequency

    @property
    def phase(self) -> torch.Tensor:
        """The global gait phase in cycles, with shape (num_envs, 1)."""
        return self._phase

    @property
    def phase_features(self) -> torch.Tensor:
        """Sine/cosine encoding of the global gait phase."""
        phase_angle = 2.0 * torch.pi * self._phase
        return torch.cat((torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1)

    @property
    def left_phase(self) -> torch.Tensor:
        """The left-leg phase, using the global phase as its reference."""
        return self._phase

    @property
    def right_phase(self) -> torch.Tensor:
        """The right-leg phase, offset from the left leg by half a cycle."""
        return torch.remainder(self._phase + 0.5, 1.0)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        normalized_frequency = torch.clamp(actions, -1.0, 1.0)
        target_frequency = self._frequency_min + 0.5 * (normalized_frequency + 1.0) * (
            self._frequency_max - self._frequency_min
        )
        self._filtered_frequency[:] = (
            self._ema_alpha * target_frequency + (1.0 - self._ema_alpha) * self._filtered_frequency
        )
        self._phase[:] = torch.remainder(self._phase + self._control_dt * self._filtered_frequency, 1.0)
        self._processed_actions[:] = self._filtered_frequency

    def apply_actions(self) -> None:
        return

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._filtered_frequency[env_ids] = self.cfg.initial_frequency
        self._phase[env_ids] = self.cfg.initial_phase % 1.0
        self._processed_actions[env_ids] = self._filtered_frequency[env_ids]
