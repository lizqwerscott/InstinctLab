"""
injector.py — dynamically swap DCMFootholdPlanner parameters at runtime.

The core idea:  FootholdProximityReward creates a DCMFootholdPlanner with
hard-coded default parameters in its __init__.  Rather than modifying the
production reward code, we use a context manager that reaches into the
live environment and hot-swaps the planner instance for the duration of
one evaluation trial.

Design decisions:
  - We replace the *entire* planner object (not individual attributes)
    so that any cached state (Sobel kernels, grid meshes, etc.) is
    correctly re-created.
  - The context manager restores the original planner on exit, even if
    an exception occurs inside the `with` block.
  - If the reward term cannot be found (e.g. task variant without
    foothold_proximity), a clear RuntimeError is raised.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from instinctlab.tasks.parkour.mdp.dcm_planner import DCMFootholdPlanner

if TYPE_CHECKING:
    from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper


class PlannerInjector:
    """Context manager: temporarily replace a reward term's DCM planner.

    Typical usage::

        params = {"alpha_pos": 1.5, "alpha_dcm": 0.3, ...}
        with PlannerInjector(env, params):
            score = run_rollout(env, policy)
        # Planner is automatically restored here.

    Attributes:
        env:  The wrapped vectorised environment.
        params:  Keyword arguments forwarded to DCMFootholdPlanner.__init__.
        _original_planner:  Reference to the planner that was active before
            ``__enter__``, so it can be restored in ``__exit__``.
        _reward_term:  The FootholdProximityReward instance whose planner we
            swapped.  ``None`` if the term was not found.
    """

    # Name of the reward term in the reward-manager config that holds the
    # foothold planner.  Must match the key in parkour_env_cfg.RewardsCfg.
    _TERM_NAME: str = "foothold_proximity"

    def __init__(
        self,
        env: "InstinctRlVecEnvWrapper",
        params: Dict[str, float],
    ) -> None:
        """Store the target environment and planner parameters.

        Args:
            env:  Wrapped vectorised environment (must contain a
                ``ManagerBasedRLEnv`` whose reward manager includes a
                ``foothold_proximity`` term).
            params:  Planner kwargs.  Only keys that match
                ``DCMFootholdPlanner.__init__`` parameters are forwarded;
                extras are silently ignored.
        """
        self._env = env
        self._params = params
        self._original_planner: DCMFootholdPlanner | None = None
        self._reward_term = None  # set in __enter__

    # ------------------------------------------------------------------
    def __enter__(self) -> "PlannerInjector":
        """Swap the foothold reward's planner for one with trial parameters.

        Returns ``self`` so the caller can inspect ``_reward_term`` if needed.

        Raises:
            RuntimeError:  If the reward term ``foothold_proximity`` is not
                found in the environment's reward manager.
        """
        # Walk through the unwrapped environment chain:
        #   InstinctRlVecEnvWrapper → (gym Env) → ManagerBasedRLEnv
        unwrapped = self._env.unwrapped

        # The reward manager stores configured terms in ``_term_cfgs`` (a dict
        # of ``RewardTermCfg`` keyed by term name) and live term instances in
        # ``_term_indices`` (term_name → index into the internal list).
        reward_mgr = unwrapped.reward_manager
        term_names = getattr(reward_mgr, "_term_names", [])

        if self._TERM_NAME not in term_names:
            raise RuntimeError(
                f"Reward term '{self._TERM_NAME}' not found in the environment's "
                f"reward manager. Available terms: {term_names}. "
                f"Make sure the task is a parkour variant with foothold_proximity."
            )

        # Locate the live term instance.
        # Isaac Lab's RewardManager stores term instances in an internal
        # list ``_terms``, indexed by their position in ``_term_names``.
        term_idx = term_names.index(self._TERM_NAME)
        term_instance = reward_mgr._terms[term_idx]
        self._reward_term = term_instance

        # Save the original planner so we can restore it on __exit__.
        self._original_planner = term_instance._planner

        # ---- Build a new planner with the trial's parameters ----
        # We forward num_envs and device from the original planner to
        # preserve the environment geometry.  All other constructor
        # keyword arguments come from the trial's parameter dict.
        planner_kwargs = {
            "num_envs": self._original_planner.num_envs,
            "device": self._original_planner.device,
            **self._params,  # trial-specific weights override defaults
        }
        term_instance._planner = DCMFootholdPlanner(**planner_kwargs)

        # ---- Also reset the per-foot cache so stale p_star values from
        #      the previous planner are not reused. ----
        # The reward term caches planned footholds in ``_p_star_cache``.
        # Clearing these forces fresh planning with the new weights.
        term_instance._p_star_cache.zero_()
        term_instance._p_star_initialized.zero_()
        term_instance._swing_planned.zero_()

        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the original planner, even if an exception occurred.

        The return value is always ``None`` (False-y) so exceptions
        propagate normally.
        """
        if self._original_planner is not None and self._reward_term is not None:
            self._reward_term._planner = self._original_planner

            # Clear caches again so the restored planner starts fresh.
            self._reward_term._p_star_cache.zero_()
            self._reward_term._p_star_initialized.zero_()
            self._reward_term._swing_planned.zero_()

        # Do not suppress exceptions.
        return None
