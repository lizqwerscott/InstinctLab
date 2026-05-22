"""Fast offline checks for TerrainAwareAmassMotion — no IsaacSim required.

Run:
    source /home/zhipengli/parkour/env_isaaclab/bin/activate
    python scripts/check_terrain_aware_motion.py

It stubs isaaclab + the AmassMotion parent so the pure-python logic of
terrain_aware_amass_motion.py can be exercised in isolation:
    - class_type is bound on cfg instances (not None)
    - __post_init__ rejects malformed / reversed / non-finite ranges
    - _build_col_to_subterrain_name matches IsaacLab curriculum placement
    - _clip_start_times_to_motion_length keeps start times in-bounds
    - _sample_env_motion_start_time routes has-range vs fallback envs correctly
    - match_scene re-samples all envs (so the very first episode is terrain-aware)
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import torch

MODULE_PATH = (
    "/home/zhipengli/parkour/InstinctLab/source/instinctlab/instinctlab/"
    "motion_reference/motion_files/terrain_aware_amass_motion.py"
)

_passed = 0
_failed = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}  {detail}")


def install_stubs() -> None:
    """Stub isaaclab + the AmassMotion parent so the module imports without IsaacSim."""
    isaaclab_mod = types.ModuleType("isaaclab")
    isaaclab_mod.__path__ = []
    sys.modules["isaaclab"] = isaaclab_mod

    stub_utils = types.ModuleType("isaaclab.utils")
    stub_utils.__path__ = []

    def configclass(cls):
        # Mimic IsaacLab @configclass: produce a dataclass, wrapping mutable defaults.
        annotations = getattr(cls, "__annotations__", {})
        for name in list(annotations.keys()):
            if not hasattr(cls, name):
                continue
            val = getattr(cls, name)
            if isinstance(val, (dict, list, set)):
                setattr(cls, name, field(default_factory=lambda v=val: type(v)(v)))
        return dataclass(cls)

    stub_utils.configclass = configclass
    sys.modules["isaaclab.utils"] = stub_utils

    stub_scene = types.ModuleType("isaaclab.scene")
    stub_scene.InteractiveScene = object
    sys.modules["isaaclab.scene"] = stub_scene

    parent_pkg = types.ModuleType("amasspkg")
    parent_pkg.__path__ = []
    sys.modules["amasspkg"] = parent_pkg

    amass_mod = types.ModuleType("amasspkg.amass_motion")

    class AmassMotion:
        """Minimal parent stub: records ratio-sampling calls so the override can be observed."""

        def __init__(self, cfg, *a, **k):
            self.cfg = cfg
            self.ratio_sampled_ids = None

        def match_scene(self, scene):
            pass

        def _sample_env_motion_start_time(self, assigned_ids):
            # record which envs the parent ratio sampler touched
            self.ratio_sampled_ids = torch.as_tensor(assigned_ids).clone()

    amass_mod.AmassMotion = AmassMotion
    sys.modules["amasspkg.amass_motion"] = amass_mod

    amass_cfg_mod = types.ModuleType("amasspkg.amass_motion_cfg")

    @dataclass
    class AmassMotionCfg:
        motion_start_from_middle_range: list = field(default_factory=lambda: [0.0, 0.0])
        motion_bin_length_s: float | None = None
        class_type: type = type(None)

    amass_cfg_mod.AmassMotionCfg = AmassMotionCfg
    sys.modules["amasspkg.amass_motion_cfg"] = amass_cfg_mod


def load_module():
    spec = importlib.util.spec_from_file_location("amasspkg.terrain_aware_amass_motion", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["amasspkg.terrain_aware_amass_motion"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_class_type(M) -> None:
    print("[test] class_type binding")
    inst = M.TerrainAwareAmassMotionCfg()
    check("instance.class_type is TerrainAwareAmassMotion", inst.class_type is M.TerrainAwareAmassMotion,
          f"got {inst.class_type}")


def test_validation(M) -> None:
    print("[test] __post_init__ range validation")
    bad_cases = {
        "reversed": {"x": (5.0, 1.0)},
        "negative": {"x": (-1.0, 2.0)},
        "nan": {"x": (float("nan"), 1.0)},
        "wrong-len": {"x": (1.0,)},
    }
    for label, bad in bad_cases.items():
        try:
            M.TerrainAwareAmassMotionCfg(subterrain_time_ranges_s=bad)
            check(f"rejects {label}", False, "no ValueError raised")
        except ValueError:
            check(f"rejects {label}", True)
    try:
        ok = M.TerrainAwareAmassMotionCfg(subterrain_time_ranges_s={"pyramid_stairs": (10.0, 10.5)})
        check("accepts valid range", ok.subterrain_time_ranges_s == {"pyramid_stairs": (10.0, 10.5)})
    except ValueError as e:
        check("accepts valid range", False, str(e))


def _fake_gen():
    class FakeSub:
        def __init__(self, p):
            self.proportion = p

    class FakeGen:
        num_cols = 20
        sub_terrains = {
            "perlin_rough": FakeSub(0.20),
            "perlin_rough_stand": FakeSub(0.20),
            "pyramid_stairs": FakeSub(0.15),
            "pyramid_stairs_tiny": FakeSub(0.15),
            "pyramid_stairs_inv": FakeSub(0.15),
            "pyramid_stairs_inv_tiny": FakeSub(0.15),
        }

    return FakeGen


def test_col_mapping(M) -> None:
    print("[test] _build_col_to_subterrain_name")
    FakeGen = _fake_gen()
    mapping = M._build_col_to_subterrain_name(FakeGen)
    check("mapping length == num_cols", len(mapping) == 20, f"got {len(mapping)}")
    from collections import Counter

    counts = Counter(mapping)
    check("perlin_rough gets 4 cols", counts["perlin_rough"] == 4, f"got {counts['perlin_rough']}")
    check("pyramid_stairs gets 3 cols", counts["pyramid_stairs"] == 3, f"got {counts['pyramid_stairs']}")
    check("all 6 subterrains present", len(counts) == 6, f"got {sorted(counts)}")

    props = np.array([0.20, 0.20, 0.15, 0.15, 0.15, 0.15])
    props = props / props.sum()
    cumsum = np.cumsum(props)
    names = list(FakeGen.sub_terrains.keys())
    expected = [names[int(np.min(np.where(c / 20 + 0.001 < cumsum)[0]))] for c in range(20)]
    check("matches IsaacLab formula", mapping == expected, f"\n   got {mapping}\n   exp {expected}")


def _make_motion_buffer(M, num_envs, motion_lengths_frames, framerate=50.0):
    """Construct a TerrainAwareAmassMotion with just enough fake state for sampling tests."""
    buf = M.TerrainAwareAmassMotion.__new__(M.TerrainAwareAmassMotion)
    buf.output_device = torch.device("cpu")
    buf.buffer_device = torch.device("cpu")
    buf.device = torch.device("cpu")
    buf.assigned_env_slice = slice(0, num_envs)
    buf._assigned_env_motion_selection = torch.arange(num_envs)
    seq = types.SimpleNamespace()
    seq.buffer_length = torch.tensor(motion_lengths_frames, dtype=torch.float)
    seq.framerate = torch.full((num_envs,), framerate)
    buf._all_motion_sequences = seq
    buf._motion_buffer_start_time_s = torch.zeros(num_envs)
    buf.ratio_sampled_ids = None
    buf.cfg = M.TerrainAwareAmassMotionCfg()  # debug_print_sampled_start defaults False
    return buf


def test_clip(M) -> None:
    print("[test] _clip_start_times_to_motion_length")
    # env0: 100 frames @ 50fps -> 2.0s ; env1: 500 frames -> 10.0s
    buf = _make_motion_buffer(M, num_envs=2, motion_lengths_frames=[100, 500], framerate=50.0)
    buf._motion_buffer_start_time_s = torch.tensor([99.0, 3.0])  # env0 way overshoots
    buf._clip_start_times_to_motion_length(torch.tensor([0, 1]))
    st = buf._motion_buffer_start_time_s
    check("overshoot env clipped below length", abs(st[0].item() - 1.98) < 1e-5, f"got {st[0].item()}")
    check("in-range env unchanged", abs(st[1].item() - 3.0) < 1e-6, f"got {st[1].item()}")
    buf._motion_buffer_start_time_s = torch.tensor([-5.0, 3.0])
    buf._clip_start_times_to_motion_length(torch.tensor([0, 1]))
    check("negative clamped to 0", buf._motion_buffer_start_time_s[0].item() == 0.0,
          f"got {buf._motion_buffer_start_time_s[0].item()}")


def test_sampling_routing(M) -> None:
    print("[test] _sample_env_motion_start_time routing")
    # All motions 500 frames @ 50fps -> 10.0s long; clip slack = 1/50 = 0.02s.
    buf = _make_motion_buffer(M, num_envs=4, motion_lengths_frames=[500, 500, 500, 500], framerate=50.0)
    buf._env_start_range_s = torch.tensor([10.0, 0.0, 4.0, 0.0])
    buf._env_end_range_s = torch.tensor([10.0, 0.0, 4.0, 0.0])  # zero-width -> deterministic
    buf._env_has_subterrain_range = torch.tensor([True, False, True, False])

    buf._sample_env_motion_start_time(torch.tensor([0, 1, 2, 3]))
    st = buf._motion_buffer_start_time_s
    check("terrain env0 sampled at 10.0 then clipped to 9.98", abs(st[0].item() - 9.98) < 1e-4,
          f"got {st[0].item()}")
    check("terrain env2 sampled at 4.0 (in range, not clipped)", abs(st[2].item() - 4.0) < 1e-5,
          f"got {st[2].item()}")
    check("fallback env1 left at 0 (parent stub no-op)", st[1].item() == 0.0, f"got {st[1].item()}")
    check("parent ratio sampler was called", buf.ratio_sampled_ids is not None)

    buf._motion_buffer_start_time_s = torch.zeros(4)
    buf._sample_env_motion_start_time(torch.tensor([2, 3]))
    st = buf._motion_buffer_start_time_s
    check("subset: env2 sampled at 4.0", abs(st[2].item() - 4.0) < 1e-5, f"got {st[2].item()}")
    check("subset: untouched env0 stays 0", st[0].item() == 0.0, f"got {st[0].item()}")

    buf2 = _make_motion_buffer(M, num_envs=2, motion_lengths_frames=[500, 500])
    buf2._sample_env_motion_start_time(torch.tensor([0, 1]))
    check("no match_scene -> no crash", True)


def test_match_scene_resample(M) -> None:
    """match_scene must re-sample all envs so the first episode is also terrain-aware,
    even though the first sampling happened during sensor init (before the startup event).
    """
    print("[test] match_scene re-samples all envs")
    buf = _make_motion_buffer(M, num_envs=4, motion_lengths_frames=[500, 500, 500, 500], framerate=50.0)
    buf.cfg = M.TerrainAwareAmassMotionCfg(
        subterrain_time_ranges_s={"pyramid_stairs": (4.0, 4.0)}
    )

    # Simulate the pre-match_scene first sampling: no _env_has_subterrain_range yet.
    buf._sample_env_motion_start_time(buf.env_ids_to_assigned_ids(None) if hasattr(buf, "env_ids_to_assigned_ids") else torch.arange(4))
    check("pre-match_scene start times are fallback (==0)",
          torch.all(buf._motion_buffer_start_time_s == 0.0).item(),
          f"got {buf._motion_buffer_start_time_s.tolist()}")

    # Build a fake scene whose terrain layout puts all 4 envs on pyramid_stairs.
    FakeGen = _fake_gen()
    # pick a column index that maps to pyramid_stairs
    mapping = M._build_col_to_subterrain_name(FakeGen)
    stairs_col = mapping.index("pyramid_stairs")

    class FakeTerrain:
        terrain_types = torch.full((4,), stairs_col, dtype=torch.long)

        class cfg:
            terrain_generator = FakeGen

    class FakeScene:
        num_envs = 4
        terrain = FakeTerrain()

    # env_ids_to_assigned_ids may not exist on the bare instance; provide it.
    if not hasattr(buf, "env_ids_to_assigned_ids"):
        buf.env_ids_to_assigned_ids = lambda env_ids: (
            torch.arange(4) if env_ids is None else torch.as_tensor(env_ids)
        )

    buf.match_scene(FakeScene())

    check("match_scene built _env_has_subterrain_range", hasattr(buf, "_env_has_subterrain_range"))
    check("all 4 envs flagged terrain-aware", bool(buf._env_has_subterrain_range.all().item()),
          f"got {buf._env_has_subterrain_range.tolist()}")
    # after re-sample, every env should sit at the configured 4.0s (in range, not clipped)
    check("match_scene re-sampled start times to 4.0",
          torch.allclose(buf._motion_buffer_start_time_s, torch.full((4,), 4.0), atol=1e-4),
          f"got {buf._motion_buffer_start_time_s.tolist()}")


def main() -> int:
    install_stubs()
    M = load_module()
    test_class_type(M)
    test_validation(M)
    test_col_mapping(M)
    test_clip(M)
    test_sampling_routing(M)
    test_match_scene_resample(M)
    print()
    print(f"=== {_passed} passed, {_failed} failed ===")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
