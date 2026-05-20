"""AmassMotion variant that constrains motion start time by each env's subterrain type.

The standard AmassMotion samples the start time uniformly from a ratio range of the motion file
(motion_start_from_middle_range). When the motion dataset contains multiple skill segments
(e.g. walk in 0-10s, stairs in 10-20s, turn in 20-30s) and the training environment lays out
heterogeneous subterrains, mixing all segments across all envs makes the AMP discriminator
learn an averaged distribution.

This buffer reads scene.terrain at startup to figure out which subterrain each env sits on,
then on each reset samples the motion start time from the absolute-second range bound to that
subterrain. Envs whose subterrain is not configured fall back to the parent ratio-based sampling
so behavior matches the vanilla AmassMotion exactly.
"""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .amass_motion import AmassMotion
from .amass_motion_cfg import AmassMotionCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

import torch


def _build_col_to_subterrain_name(terrain_gen_cfg) -> list[str]:
    """Mirror IsaacLab's curriculum subterrain placement to map column index to sub_terrain name.

    Matches the logic in IsaacLab's TerrainGenerator._generate_curriculum_terrains:
        sub_index = min(where(col_idx / num_cols + 0.001 < cumsum(proportions))[0])
    """
    names = list(terrain_gen_cfg.sub_terrains.keys())
    proportions = np.array([c.proportion for c in terrain_gen_cfg.sub_terrains.values()], dtype=np.float64)
    proportions = proportions / proportions.sum()
    cumsum = np.cumsum(proportions)
    num_cols = terrain_gen_cfg.num_cols
    col_to_name = []
    for col in range(num_cols):
        sub_idx = int(np.min(np.where(col / num_cols + 0.001 < cumsum)[0]))
        col_to_name.append(names[sub_idx])
    return col_to_name


class TerrainAwareAmassMotion(AmassMotion):
    """AmassMotion that picks start time per env according to its subterrain.

    Forward-declared before the cfg so `class_type: type = TerrainAwareAmassMotion`
    is bound at dataclass-field-default time.
    """

    cfg: "TerrainAwareAmassMotionCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        assert self.cfg.motion_bin_length_s is None, (
            "TerrainAwareAmassMotion does not support motion_bin_length_s; the bin sampling path"
            " in the parent class would override the terrain-aware start time."
        )

    """
    Scene matching: cache env -> subterrain mask at startup.
    """

    def match_scene(self, scene: "InteractiveScene") -> None:
        """Build per-env start-time ranges (in seconds) from the scene's subterrain layout.

        Envs whose subterrain has no entry in `subterrain_time_ranges_s` are marked
        `has_range = False` and at reset will route back through the parent ratio sampler.
        """
        super().match_scene(scene)

        # Default everyone to "no terrain-aware range"; populate below if applicable.
        self._env_start_range_s = torch.zeros(scene.num_envs, device=self.device)
        self._env_end_range_s = torch.zeros(scene.num_envs, device=self.device)
        self._env_has_subterrain_range = torch.zeros(scene.num_envs, dtype=torch.bool, device=self.device)

        terrain_importer = scene.terrain
        terrain_gen_cfg = terrain_importer.cfg.terrain_generator
        if terrain_gen_cfg is None:
            print("[TerrainAwareAmassMotion] terrain has no generator (plane?); using parent ratio sampler for all envs.")
            return

        terrain_types = getattr(terrain_importer, "terrain_types", None)
        if terrain_types is None:
            print(
                "[TerrainAwareAmassMotion] terrain_importer.terrain_types is None (non-curriculum mode); "
                "using parent ratio sampler for all envs."
            )
            return

        col_to_name = _build_col_to_subterrain_name(terrain_gen_cfg)

        # Vectorized fill via a per-col-name LUT, no python loop over envs.
        for col_idx, name in enumerate(col_to_name):
            if name not in self.cfg.subterrain_time_ranges_s:
                continue
            s, e = self.cfg.subterrain_time_ranges_s[name]
            mask = (terrain_types == col_idx).to(self.device)
            self._env_start_range_s[mask] = float(s)
            self._env_end_range_s[mask] = float(e)
            self._env_has_subterrain_range[mask] = True

        # Sanity-summary line, useful when adding new subterrain entries.
        unique_types = torch.unique(terrain_types).tolist()
        summary = {col_to_name[int(t)]: int((terrain_types == t).sum().item()) for t in unique_types}
        configured = {n: cnt for n, cnt in summary.items() if n in self.cfg.subterrain_time_ranges_s}
        fallback = {n: cnt for n, cnt in summary.items() if n not in self.cfg.subterrain_time_ranges_s}
        print(
            f"[TerrainAwareAmassMotion] env counts by subterrain — configured: {configured}, "
            f"fallback-to-ratio: {fallback}"
        )

        # The first start-time sampling happens during sensor init (set_env_ids_assignments),
        # which runs before this startup event. That first sample therefore used parent ratio
        # fallback. Re-sample every assigned env now so the very first episode also gets
        # terrain-aware start times.
        self._sample_env_motion_start_time(self.env_ids_to_assigned_ids(None))

    """
    Override start-time sampling.
    """

    def _sample_env_motion_start_time(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Per-env start time: terrain-aware envs sample from absolute-second range,
        others go through the parent ratio-based sampler. Both paths are clipped to
        the assigned motion's actual length so frame indices stay in bounds.
        """
        if len(assigned_ids) == 0:
            return

        # Step 1: parent ratio sampling fills _motion_buffer_start_time_s for all assigned_ids.
        super()._sample_env_motion_start_time(assigned_ids)

        if not hasattr(self, "_env_has_subterrain_range"):
            # Expected on the first sampling call (sensor init runs before the
            # match_motion_ref_with_scene startup event). match_scene() re-samples all envs
            # afterwards, so this first fallback is harmless. If match_scene never runs,
            # every reset keeps landing here — the WARNING below makes that visible.
            if not getattr(self, "_warned_no_match_scene", False):
                print(
                    "[TerrainAwareAmassMotion] match_scene not done yet at first sampling "
                    "(expected during sensor init); will re-sample after the startup event. "
                    "If this WARNING repeats after training starts, the "
                    "match_motion_ref_with_scene startup event is not registered."
                )
                self._warned_no_match_scene = True
            self._clip_start_times_to_motion_length(assigned_ids)
            return

        # Step 2: overwrite the entries with a configured subterrain range.
        assigned_ids_t = torch.as_tensor(assigned_ids, device=self.output_device, dtype=torch.long)
        env_ids_abs = assigned_ids_t + self.assigned_env_slice.start

        has_range = self._env_has_subterrain_range[env_ids_abs]
        if has_range.any():
            local_idx = torch.nonzero(has_range, as_tuple=False).flatten()
            env_ids_with_range = env_ids_abs[local_idx]

            starts = self._env_start_range_s[env_ids_with_range]
            ends = self._env_end_range_s[env_ids_with_range]
            rnd = torch.rand(len(local_idx), device=self.output_device)
            sampled = starts + rnd * (ends - starts)

            target_local = assigned_ids_t[local_idx].to(self.buffer_device)
            self._motion_buffer_start_time_s[target_local] = sampled.to(self.buffer_device)

        # Step 3: clip everyone's start time to the actual motion length so that even
        # ranges that overshoot a particular motion never produce out-of-bounds frames.
        self._clip_start_times_to_motion_length(assigned_ids)

    def _clip_start_times_to_motion_length(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Clamp _motion_buffer_start_time_s[assigned_ids] to [0, motion_length - eps].

        Keeps a small epsilon below the end so fill_init_reference_state never lands on
        the last frame (where frame_selection == buffer_length triggers the validity=False path).
        """
        assigned_ids_t = torch.as_tensor(assigned_ids, device=self.buffer_device, dtype=torch.long)
        motion_ids = self._assigned_env_motion_selection[assigned_ids_t]
        max_len_s = (
            self._all_motion_sequences.buffer_length[motion_ids].to(torch.float)
            / self._all_motion_sequences.framerate[motion_ids].to(torch.float)
        )
        # Keep one frame of slack so frame_selection stays < buffer_length on the first read.
        framerate = self._all_motion_sequences.framerate[motion_ids].to(torch.float)
        slack = 1.0 / torch.clamp(framerate, min=1.0)
        upper = torch.clamp(max_len_s - slack, min=0.0)  # per-env upper bound (tensor)
        # torch.clamp does not accept scalar-min + tensor-max together; clamp in two steps.
        clipped = torch.clamp(self._motion_buffer_start_time_s[assigned_ids_t], min=0.0)
        self._motion_buffer_start_time_s[assigned_ids_t] = torch.minimum(clipped, upper)


@configclass
class TerrainAwareAmassMotionCfg(AmassMotionCfg):
    """Configuration for TerrainAwareAmassMotion.

    `subterrain_time_ranges_s` keys must match keys in
    `scene.terrain.cfg.terrain_generator.sub_terrains`. Values are absolute-second
    (start, end) ranges from which the motion start time will be uniformly sampled
    for envs sitting on that subterrain. Envs on subterrains not in the dict fall
    back to the parent class' ratio-based sampling using `motion_start_from_middle_range`.

    Recommended to write narrow ranges that already account for episode length, e.g.
    if the stairs segment is 10-20s and episode_length_s is ~10s, use (10.0, 10.5)
    instead of (10.0, 20.0) so the reference does not run out of the segment mid-rollout.
    """

    class_type: type = TerrainAwareAmassMotion

    subterrain_time_ranges_s: dict[str, tuple[float, float]] = {}
    """Mapping from sub_terrain name to (start_s, end_s). Inclusive on both ends."""

    def __post_init__(self) -> None:
        # Parent AmassMotionCfg has no __post_init__ at the time of writing, but guard anyway.
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        for name, rng in self.subterrain_time_ranges_s.items():
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                raise ValueError(
                    f"subterrain_time_ranges_s[{name!r}] must be a (start_s, end_s) pair, got {rng!r}"
                )
            s, e = rng
            if not (math.isfinite(s) and math.isfinite(e)):
                raise ValueError(f"subterrain_time_ranges_s[{name!r}] must be finite, got ({s}, {e})")
            if s < 0 or e < s:
                raise ValueError(
                    f"subterrain_time_ranges_s[{name!r}] must satisfy 0 <= start <= end, got ({s}, {e})"
                )
