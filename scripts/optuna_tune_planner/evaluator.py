"""
evaluator.py — lightweight rollout evaluator for DCM planner parameters.

This module is the bridge between Optuna and the Isaac Lab simulation.
It creates the environment **once** (expensive — involves USD stage loading,
physics initialisation, and GPU buffer allocation), then re-uses it across
all Optuna trials.  For each trial the planner parameters are hot-swapped
via ``PlannerInjector``.

Key design constraint:  this module must be imported *after* the Isaac Sim
``AppLauncher`` has started the simulation app, because the environment
constructor calls into Omniverse APIs.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import gymnasium as gym

# Isaac Lab imports — available only after AppLauncher is running.
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_rotate_inverse

# InstinctLab imports
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

from scripts.optuna_tune_planner.config import DEFAULT_PARAMS, EvalConfig
from scripts.optuna_tune_planner.injector import PlannerInjector
from scripts.optuna_tune_planner.metrics import MetricsAccumulator


class PlannerEvaluator:
    """Creates an evaluation environment, loads a frozen policy, and scores
    planner parameter sets via short rollouts.

    **Lifecycle**::

        evaluator = PlannerEvaluator(cfg, task_name, checkpoint_path)
        score = evaluator.evaluate(params_dict)   # called by Optuna
        evaluator.close()                          # release GPU resources
    """

    def __init__(
        self,
        cfg: EvalConfig,
        task_name: str,
        checkpoint_path: str,
        env_cfg=None,  # pre-built config avoids parse_env_cfg recursion
    ) -> None:
        """Initialise the environment and load the frozen policy.

        Args:
            cfg:  Evaluation configuration.
            task_name:  Gym task id.
            checkpoint_path:  Path to a .pt checkpoint.
            env_cfg:  Optional pre-built env config.  If None, parsed from
                task_name (which may trigger recursion on some versions).
        """
        self._cfg = cfg
        self._device = cfg.device

        self._env = self._create_env(task_name)
        self._policy = self._load_policy(checkpoint_path)
        self._setup_terrain_mapping()

    # ==================================================================
    # Public: evaluate one parameter set
    # ==================================================================

    def evaluate(self, params: Dict[str, float]) -> Tuple[float, float]:
        """Score one set of planner parameters.

        Runs ``num_repeat`` independent rollouts and returns the mean and
        standard deviation of the composite score.

        Args:
            params:  Dictionary of ``DCMFootholdPlanner`` constructor kwargs.
                Only keys present in ``SEARCH_SPACE`` are used; extras are
                silently ignored.

        Returns:
            (mean_score, std_score) — both floats.  Higher is better.
        """
        trial_scores: List[float] = []

        for rep in range(self._cfg.num_repeat):
            # The PlannerInjector context manager handles:
            #   1. Swapping the planner  (__enter__)
            #   2. Running the rollout
            #   3. Restoring the original planner (__exit__)
            with PlannerInjector(self._env, params):
                score = self._run_single_rollout(seed_offset=rep)
                trial_scores.append(score)

        mean_score = float(np.mean(trial_scores))
        std_score = float(np.std(trial_scores)) if len(trial_scores) > 1 else 0.0
        return mean_score, std_score

    # ==================================================================
    # Public: baseline comparison
    # ==================================================================

    def evaluate_baseline(self, num_repeat: int = 5) -> Tuple[float, float]:
        """Evaluate the current (default) planner parameters for A/B reference.

        Args:
            num_repeat:  Number of independent rollouts.

        Returns:
            (mean_score, std_score) for the default parameters.
        """
        return self.evaluate(DEFAULT_PARAMS)

    # ==================================================================
    # Public: cleanup
    # ==================================================================

    def close(self) -> None:
        """Release the environment and GPU resources."""
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()

    # ==================================================================
    # Environment creation
    # ==================================================================

    def _create_env(self, task_name: str, env_cfg=None) -> InstinctRlVecEnvWrapper:
        """Build a lightweight evaluation environment.

        Monkey-patches ``configclass._validate`` to a no-op before calling
        ``parse_env_cfg`` + ``gym.make``, then restores it.  This is the
        only reliable way to avoid infinite recursion in Isaac Lab's
        configclass validation.
        """
        from isaaclab.utils import configclass as _cc

        _orig_validate = getattr(_cc, '_validate', None)
        if _orig_validate is not None:
            _cc._validate = lambda obj, prefix='': ([], [])

        try:
            from isaaclab_tasks.utils import parse_env_cfg
            env_cfg = parse_env_cfg(
                task_name, device=self._device, num_envs=self._cfg.num_envs,
            )
            env_cfg.scene.num_envs = self._cfg.num_envs
            env_cfg.episode_length_s = 20.0

            if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
                env_cfg.curriculum = None
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None
            if hasattr(env_cfg.events, "physics_material"):
                env_cfg.events.physics_material = None

            for field_name in env_cfg.rewards.__dataclass_fields__:
                if field_name == "rewards":
                    rewards_cfg = getattr(env_cfg.rewards, field_name)
                    for rew_field in rewards_cfg.__dataclass_fields__:
                        rew_term = getattr(rewards_cfg, rew_field)
                        if hasattr(rew_term, "params") and rew_term.params is not None:
                            if "debug_vis" in rew_term.params:
                                rew_term.params["debug_vis"] = False

            env = gym.make(task_name, cfg=env_cfg)
            env = InstinctRlVecEnvWrapper(env)
        finally:
            if _orig_validate is not None:
                _cc._validate = _orig_validate

        return env

    # ==================================================================
    # Policy loading
    # ==================================================================

    def _load_policy(self, checkpoint_path: str):
        """Load a trained policy checkpoint and return an inference function.

        The returned callable maps observations → actions **without** any
        exploration noise (deterministic mode).

        Args:
            checkpoint_path:  Path to a ``.pt`` file saved by ``OnPolicyRunner``.

        Returns:
            A callable ``policy(obs: torch.Tensor) -> torch.Tensor``.
        """
        from instinct_rl.runners import OnPolicyRunner

        # We use a minimal runner config — only the algorithm class name and
        # device matter for loading the network weights.  The actual training
        # hyper-parameters (lr, gamma, etc.) are irrelevant for inference.
        from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

        # Build a scratch config by introspecting the environment.
        obs_dict, _ = self._env.get_observations()
        agent_cfg = InstinctRlOnPolicyRunnerCfg()
        agent_cfg.device = self._device

        # Create runner and load checkpoint.
        # ``log_dir=None`` suppresses TensorBoard logging.
        runner = OnPolicyRunner(
            self._env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=self._device,
        )
        runner.load(checkpoint_path)

        # Return deterministic inference policy.
        return runner.get_inference_policy(device=self._device)

    # ==================================================================
    # Rollout execution
    # ==================================================================

    def _run_single_rollout(self, seed_offset: int = 0) -> float:
        """Execute one rollout and return the composite score.

        The environment is reset at the beginning; the frozen policy runs
        in ``torch.inference_mode()`` for the full ``rollout_steps`` duration.

        Args:
            seed_offset:  Added to the base seed so repeated rollouts of the
                same parameter set explore different random initial states.

        Returns:
            Composite scalar score (higher is better).
        """
        # ---- Reset the environment ----
        obs, _ = self._env.get_observations()

        # ---- Create metrics accumulator ----
        accum = MetricsAccumulator(self._cfg)

        # ---- Rollout loop ----
        for step in range(self._cfg.rollout_steps):
            # Inference — no gradients, no exploration noise.
            with torch.inference_mode():
                actions = self._policy(obs)

            # Step the simulation.
            obs, rewards, dones, infos = self._env.step(actions)

            # ---- Extract per-step metrics ----
            # 1. Foothold proximity reward:
            #    The reward manager stores per-term values; we extract the
            #    foothold_proximity term's contribution.
            foothold_r = self._get_foothold_reward_per_env()

            # 2. Tracking error: L2 norm of (cmd_vel - actual_vel) in body frame.
            tracking_err = self._get_tracking_error_per_env()

            # 3. Foot slip: mean sliding velocity of feet in contact.
            slip = self._get_foot_slip_per_env()

            # 4. Terrain IDs for per-terrain breakdown.
            terrain_ids = self._get_terrain_ids()

            # ---- Feed the accumulator ----
            accum.update(
                foothold_reward=foothold_r,
                tracking_error=tracking_err,
                foot_slip=slip,
                done_mask=dones,
                terrain_ids=terrain_ids,
            )

        # ---- Compute the final score ----
        return accum.compute_score()

    # ==================================================================
    # Per-step metric extractors
    # ==================================================================

    def _get_foothold_reward_per_env(self) -> np.ndarray:
        """Return the per-environment foothold_proximity reward for the
        current step, as a numpy array of shape ``(num_envs,)``.

        We access the reward manager's internal ``_term_rewards`` dict,
        which maps term name → ``Tensor(num_envs,)`` of this step's values.
        """
        unwrapped = self._env.unwrapped
        reward_mgr = unwrapped.reward_manager

        # ``_term_rewards`` is populated by ``RewardManager.compute()`` each step.
        term_rewards = getattr(reward_mgr, "_term_rewards", None)
        if term_rewards is not None and "foothold_proximity" in term_rewards:
            raw = term_rewards["foothold_proximity"]
            if isinstance(raw, torch.Tensor):
                return raw.cpu().numpy()
        # Fallback: return zeros (should not normally happen).
        return np.zeros(unwrapped.num_envs, dtype=np.float32)

    def _get_tracking_error_per_env(self) -> np.ndarray:
        """Compute L2 velocity tracking error (body-frame x-y plane).

        command  = commanded lin_vel in body frame   (from command manager)
        actual   = root_lin_vel_b in body frame       (from asset data)
        error    = ‖cmd_xy − actual_xy‖₂
        """
        unwrapped = self._env.unwrapped
        asset = unwrapped.scene["robot"]

        # Commanded velocity: shape (num_envs, 3) in body frame.
        cmd_vel = unwrapped.command_manager.get_command("base_velocity")
        if isinstance(cmd_vel, torch.Tensor):
            cmd_xy = cmd_vel[:, :2]  # only horizontal components
        else:
            return np.zeros(unwrapped.num_envs, dtype=np.float32)

        # Actual velocity: shape (num_envs, 3) in body frame.
        actual_vel = asset.data.root_lin_vel_b[:, :2]

        error = torch.norm(cmd_xy - actual_vel, dim=1)
        return error.cpu().numpy()

    def _get_foot_slip_per_env(self) -> np.ndarray:
        """Estimate foot slip as the horizontal speed of foot links when
        they are in contact with the ground.

        For each foot (left / right ankle_roll_link), if the contact force
        exceeds a threshold, the slip speed is the L2 norm of the foot's
        linear velocity in the world x-y plane.  The per-environment value
        is the mean slip across both feet.
        """
        unwrapped = self._env.unwrapped
        asset = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene.sensors["contact_forces"]

        # Foot body indices — resolve from the asset's body_names.
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_indices: List[int] = []
        for name in foot_names:
            if name in asset.data.body_names:
                foot_indices.append(asset.data.body_names.index(name))

        if len(foot_indices) == 0:
            return np.zeros(unwrapped.num_envs, dtype=np.float32)

        # Contact force magnitude: use the most recent history frame.
        net_forces = contact_sensor.data.net_forces_w_history  # (N, hist, n_bodies)
        contact_force = torch.norm(net_forces[:, -1, :], dim=-1)  # (N, n_bodies)
        contact_threshold = 1.0  # N

        slip = torch.zeros(unwrapped.num_envs, device=self._device)
        valid_count = 0
        for idx in foot_indices:
            in_contact = contact_force[:, idx] > contact_threshold
            if in_contact.any():
                # Foot velocity in world frame, take x-y components.
                foot_vel = asset.data.body_lin_vel_w[:, idx, :2]  # (N, 2)
                foot_slip_speed = torch.norm(foot_vel, dim=1)     # (N,)
                slip = slip + foot_slip_speed * in_contact.float()
                valid_count += 1

        if valid_count > 0:
            slip = slip / valid_count
        return slip.cpu().numpy()

    def _get_terrain_ids(self) -> Optional[np.ndarray]:
        """Return the terrain-type index for each environment.

        The terrain importer assigns each environment a column index in the
        terrain grid, which maps to a sub-terrain type.  These indices are
        available via ``terrain.terrain_types``.
        """
        unwrapped = self._env.unwrapped
        if "terrain" not in unwrapped.scene:
            return None
        terrain = unwrapped.scene["terrain"]
        if hasattr(terrain, "terrain_types"):
            tt = terrain.terrain_types
            if isinstance(tt, torch.Tensor):
                return tt.cpu().numpy()
        return None

    # ==================================================================
    # Terrain mapping (index → name)
    # ==================================================================

    def _setup_terrain_mapping(self) -> None:
        """Build a mapping from terrain column index to sub-terrain name.

        The terrain generator divides columns among sub-terrain types
        proportionally.  We reconstruct this mapping so the metrics
        accumulator can group statistics by terrain name.
        """
        unwrapped = self._env.unwrapped
        if "terrain" not in unwrapped.scene:
            return

        terrain = unwrapped.scene["terrain"]
        cfg = getattr(terrain.cfg, "terrain_generator", None)
        if cfg is None:
            return

        sub_names = list(cfg.sub_terrains.keys())
        proportions = np.array(
            [cfg.sub_terrains[n].proportion for n in sub_names],
            dtype=np.float64,
        )
        proportions /= np.sum(proportions)
        cumsum = np.cumsum(proportions)

        mapping: Dict[int, str] = {}
        for col in range(cfg.num_cols):
            idx = int(np.min(np.where(col / cfg.num_cols + 0.001 < cumsum)[0]))
            if idx < len(sub_names):
                mapping[col] = sub_names[idx]

        MetricsAccumulator.set_terrain_mapping(mapping)
