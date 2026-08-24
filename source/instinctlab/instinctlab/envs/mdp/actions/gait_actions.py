from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import GaitFrequencyActionCfg


class GaitFrequencyAction(ActionTerm):
    """Map normalized policy actions to a filtered gait frequency, single-support
    ratio and gait phase.

    The policy outputs two normalized actions in [-1, 1]:

    * frequency: linearly scaled into ``frequency_range`` and EMA-smoothed.
    * ratio: single-support ratio r = T_SS/(T_SS+T_DS), linearly scaled into
      ``ratio_range`` and EMA-smoothed. The per-leg swing window on the phase
      axis is [0, r/2); the gap between the two legs' windows is the
      double-support phase (r = 1 recovers pure alternation).

    The phase advances at the stride frequency (one full cycle per two steps,
    right-leg phase offset by 0.5). The term has no physical effect
    (``apply_actions`` is a no-op); it only exposes the clock state to
    observations and rewards.
    """

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
        # ---- Ratio-knob validation (Egle-style single-support ratio) ----
        ratio_min, ratio_max = cfg.ratio_range
        # 允许 min == max: 课程阶段 1 用 ratio_range=(1.0, 1.0) 锁死 r, 等价纯交替步态
        if not 0.0 < ratio_min <= ratio_max <= 1.0:
            raise ValueError(f"ratio_range must satisfy 0 < min <= max <= 1, got {cfg.ratio_range}")
        if not 0.0 < cfg.ratio_ema_alpha <= 1.0:
            raise ValueError(f"ratio_ema_alpha must be in (0, 1], got {cfg.ratio_ema_alpha}")
        if not ratio_min <= cfg.initial_ratio <= ratio_max:
            raise ValueError(
                f"initial_ratio must be within ratio_range, got {cfg.initial_ratio} and {cfg.ratio_range}"
            )

        self._frequency_min = frequency_min
        self._frequency_max = frequency_max
        self._ema_alpha = cfg.ema_alpha
        self._ratio_min = ratio_min
        self._ratio_max = ratio_max
        self._ratio_ema_alpha = cfg.ratio_ema_alpha
        self._control_dt = env.step_dt
        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._filtered_frequency = torch.full(
            (env.num_envs, 1),
            cfg.initial_frequency,
            device=self.device,
            dtype=torch.float32,
        )
        self._filtered_ratio = torch.full(
            (env.num_envs, 1),
            cfg.initial_ratio,
            device=self.device,
            dtype=torch.float32,
        )
        self._phase = torch.full(
            (env.num_envs, 1),
            cfg.initial_phase % 1.0,
            device=self.device,
            dtype=torch.float32,
        )
        self._processed_actions[:] = torch.cat((self._filtered_frequency, self._filtered_ratio), dim=-1)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw policy actions before frequency/ratio processing."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The EMA-filtered actions exposed to the action manager."""
        return self._processed_actions

    @property
    def filtered_frequency(self) -> torch.Tensor:
        """The EMA-filtered stride frequency in Hz, with shape (num_envs, 1)."""
        return self._filtered_frequency

    @property
    def filtered_ratio(self) -> torch.Tensor:
        """The EMA-filtered single-support ratio r = T_SS/(T_SS+T_DS), shape (num_envs, 1)."""
        return self._filtered_ratio

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

    @property
    def swing_window(self) -> torch.Tensor:
        """Per-leg swing windows on the phase axis, with shape (num_envs, 2).

        Leg i may swing while its own phase lies in [0, r/2). Since the two
        windows [0, r/2) and [0.5, 0.5+r/2) do not overlap for r < 1, the phase
        gaps between them are the double-support intervals.
        """
        half_ratio = 0.5 * self._filtered_ratio  # (N, 1)
        left_phi = self._phase
        right_phi = torch.remainder(self._phase + 0.5, 1.0)
        return torch.cat((left_phi < half_ratio, right_phi < half_ratio), dim=-1)

    @property
    def ss_ds_sign(self) -> torch.Tensor:
        """Phase sign with shape (num_envs, 1): +1 while a leg is inside its swing
        window (single support), -1 in the double-support gaps (cf. Egle 2024)."""
        return torch.where(self.swing_window().any(dim=-1, keepdim=True), 1.0, -1.0)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        normalized_frequency = torch.clamp(actions[:, 0:1], -1.0, 1.0)
        normalized_ratio = torch.clamp(actions[:, 1:2], -1.0, 1.0)
        target_frequency = self._frequency_min + 0.5 * (normalized_frequency + 1.0) * (
            self._frequency_max - self._frequency_min
        )
        target_ratio = self._ratio_min + 0.5 * (normalized_ratio + 1.0) * (
            self._ratio_max - self._ratio_min
        )
        self._filtered_frequency[:] = (
            self._ema_alpha * target_frequency + (1.0 - self._ema_alpha) * self._filtered_frequency
        )
        self._filtered_ratio[:] = (
            self._ratio_ema_alpha * target_ratio
            + (1.0 - self._ratio_ema_alpha) * self._filtered_ratio
        )
        self._phase[:] = torch.remainder(self._phase + self._control_dt * self._filtered_frequency, 1.0)
        self._processed_actions[:] = torch.cat((self._filtered_frequency, self._filtered_ratio), dim=-1)

    def apply_actions(self) -> None:
        return

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._filtered_frequency[env_ids] = self.cfg.initial_frequency
        self._filtered_ratio[env_ids] = self.cfg.initial_ratio
        self._phase[env_ids] = self.cfg.initial_phase % 1.0
        self._processed_actions[env_ids] = torch.cat(
            (self._filtered_frequency[env_ids], self._filtered_ratio[env_ids]), dim=-1
        )
