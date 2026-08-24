from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import gait_actions, joint_actions


@configclass
class ActionOverridenJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the action overridden delayed joint position action term.

    See :class:`ActionOverridenointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.ActionOverridenJointPositionAction

    asset_cfg: SceneEntityCfg = MISSING
    """Whether to override the action with the delayed action. Defaults to False."""

    override_value: float = 0.0
    """Delay in frames before the action is overridden. Defaults to 0."""


@configclass
class GaitFrequencyActionCfg(ActionTermCfg):
    """Configuration for the gait-frequency action term."""

    class_type: type[ActionTerm] = gait_actions.GaitFrequencyAction

    frequency_range: tuple[float, float] = (0.5, 2.0)
    """Minimum and maximum feasible gait frequency in Hz."""

    ema_alpha: float = 0.1
    """EMA update coefficient applied to the scaled frequency action."""

    initial_frequency: float = 1.0
    """Frequency used before the first policy action and after reset."""

    initial_phase: float = 0.0
    """Global gait phase used after reset, represented in cycles in [0, 1)."""

    frequency_nom: float = 1.0
    """Nominal stride frequency anchoring the gait_freq_anchor reward."""

    ratio_range: tuple[float, float] = (0.55, 0.95)
    """Minimum and maximum single-support ratio r = T_SS/(T_SS+T_DS). r < 1 leaves a
    double-support gap between the two swing windows (r = 1 recovers pure alternation)."""

    ratio_ema_alpha: float = 0.1
    """EMA update coefficient applied to the scaled ratio action."""

    initial_ratio: float = 0.75
    """Single-support ratio used before the first policy action and after reset."""

    ratio_nom: float = 0.75
    """Nominal ratio anchoring the ss_ratio_anchor reward (Egle Table I: lambda=5)."""
