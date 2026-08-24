from __future__ import annotations

import math
import os
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
import instinctlab.terrains as terrain_gen
from instinctlab.assets.unitree_g1 import beyondmimic_action_scale
from instinctlab.managers import MultiRewardCfg
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.sensors import (
    Grid3dPointsGeneratorCfg,
    NoisyGroupedRayCasterCameraCfg,
    VolumePointsCfg,
)
from instinctlab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporterCfg
from instinctlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    DepthSteroNoiseCfg,
    GaussianBlurNoiseCfg,
    RandomGaussianNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    ParametricDepthNoiseCfg,
)

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

##
# Scene definition
##
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "perlin_rough": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.20,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "perlin_rough_stand": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.20,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.16),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.16),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "up_down": terrain_gen.PerlinStairsUpDownWithWallsTerrainCfg(
            proportion=0.20,
            per_step_height=[0.05, 0.16],
            per_step_width=2.0,
            per_step_length=(0.3),
            num_steps=(4, 8),
            platform_length=2.5,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            side_wall_prob=[0.8, 0.8],
            side_wall_height=5.0,
            side_wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
        "down_up": terrain_gen.PerlinStairsDownUpTerrainCfg(
            proportion=0.20,
            per_step_height=[0.05, 0.16],
            per_step_width=2.0,
            per_step_length=(0.3),
            num_steps=(4, 8),
            platform_length=2.5,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(0.0, 0.0),
                ),
            },
        ),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.05,
                min_points=2,
            ),
        },
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.045,
            x_max=0.14,
            x_num=14,
            y_min=-0.04,
            y_max=0.04,
            y_num=7,
            z_min=-0.05,
            z_max=0.0,
            z_num=2,
        ),
        debug_vis=False,
    )
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(  # G1 Robot head camera nominal pose
            pos=(
                0.0487988662332928,
                0.01,
                0.4378029937970051,
            ),
            rot=(
                0.9135367613482678,
                0.004363309284746571,
                0.4067366430758002,
                0.0,
            ),
            convention="world",
        ),
        min_distance=0.1,
        # noise
        noise_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "parametric_depth_noise": ParametricDepthNoiseCfg(
                focal_length=31.35,
                baseline=0.05,
                min_depth=0.2,
                max_depth=5,
            ),
            # "gaussian_noise": RangeBasedGaussianNoiseCfg(noise_std = 0.02, min_value = 0.2, max_value = 1.5),
            # "stereo_failure": DepthSteroNoiseCfg(
            #     stero_far_distance=3.0,
            #     stero_min_distance=0.12,
            #     stero_far_noise_std=0.08,
            #     stero_near_noise_std=0.02,
            #     stero_full_block_artifacts_prob=0.002,
            #     stero_full_block_values=[0.0, 0.25, 0.5, 1.0, 3.0],
            #     stero_full_block_height_mean_std=[18, 2.0],
            #     stero_full_block_width_mean_std=[3, 0.5],
            #     stero_half_block_spark_prob=0.05,
            #     stero_half_block_value=3000,
            # ),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            # "random_gaussian_noise": RandomGaussianNoiseCfg(noise_mean=0.0, noise_std=1, probability=0.05),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        image_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={
            "distance_to_image_plane_noised": 37,
            "distance_to_image_plane_handled": 37,
        },
    )
    heightmap_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # attach_yaw_only=True,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    heightmap = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(1.80, 1.20),
            direction=(0.0, 0.0, -1.0),
        ),
        debug_vis=False,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    motion_reference: MotionReferenceManagerCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        gait_frequency = ObsTerm(
            func=mdp.gait_frequency,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_ratio = ObsTerm(
            func=mdp.gait_ratio,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_ss_ds_sign = ObsTerm(
            func=mdp.gait_ss_ds_sign,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=8,
            flatten_history_dim=True,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True)
        height_scan = ObsTerm(
            func=mdp.height_scan_feat,
            params={"sensor_cfg": SceneEntityCfg("heightmap_scanner")},
            history_length=4,
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=8, flatten_history_dim=True)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
            scale=0.25,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=8, flatten_history_dim=True)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=8,
            flatten_history_dim=True,
            params={"command_name": "base_velocity"},
            noise=None,
        )
        gait_frequency = ObsTerm(
            func=mdp.gait_frequency,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_ratio = ObsTerm(
            func=mdp.gait_ratio,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        gait_ss_ds_sign = ObsTerm(
            func=mdp.gait_ss_ds_sign,
            history_length=8,
            flatten_history_dim=True,
            params={"action_name": "gait_frequency"},
            noise=None,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=8, flatten_history_dim=True)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True)
        height_scan = ObsTerm(
            func=mdp.height_scan_feat,
            params={"sensor_cfg": SceneEntityCfg("heightmap_scanner")},
            history_length=4,
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class AmpPolicyStateObsCfg(ObsGroup):
        concatenate_terms = False
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            history_length=10,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    preserve_order=True,
                ),
            },
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    @configclass
    class AmpReferenceStateObsCfg(ObsGroup):
        concatenate_terms = False
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_reference_as_state,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
            history_length=10,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_reference_as_state,
            scale=0.05,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_reference_as_state,
            history_length=10,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg(name="motion_reference"),
            },
        )

    # observation group
    policy: PolicyCfg = PolicyCfg()
    # critic group
    critic: CriticCfg = CriticCfg()
    # AMP training groups
    amp_policy: AmpPolicyStateObsCfg = AmpPolicyStateObsCfg()
    amp_reference: AmpReferenceStateObsCfg = AmpReferenceStateObsCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=beyondmimic_action_scale,
        use_default_offset=True,
    )
    gait_frequency = mdp.GaitFrequencyActionCfg(
        asset_name="robot",
        # 口径: φ 一周 = 步幅(两步)。T_swing=0.45 时交替步态 f≈1.11 Hz,
        # 加 T_ds=0.15 后 f≈0.83 Hz, (0.7, 1.3) 覆盖两者。
        frequency_range=(0.7, 1.3),
        frequency_nom=1.0,
        ema_alpha=0.2,
        initial_frequency=1.0,
        initial_phase=0.0,
        ratio_range=(0.55, 0.95),
        ratio_nom=0.75,
        ratio_ema_alpha=0.2,
        initial_ratio=0.75,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        ranges=mdp.PoseVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
        random_velocity_terrain=["perlin_rough_stand"],
        velocity_ranges={
            "perlin_rough": {
                "lin_vel_x": (0.45, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "perlin_rough_stand": {
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            },
            "pyramid_stairs": {
                "lin_vel_x": (0.45, 0.6),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "pyramid_stairs_inv": {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "up_down": {
                "lin_vel_x": (0.45, 0.6),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "down_up": {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


@configclass
class G1Rewards:
    """Reward terms for the MDP."""

    # Task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.3,
        params={"command_name": "base_velocity", "offset": 4.0},
    )

    # Regularization rewards
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "vel_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    gait_freq_anchor = RewTerm(
        func=mdp.gait_freq_anchor_reward, weight=2.0, params={"action_name": "gait_frequency"}
    )
    ss_ratio_anchor = RewTerm(
        func=mdp.ss_ratio_anchor_reward, weight=1.0, params={"action_name": "gait_frequency"}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    )
    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_at_plane = RewTerm(
        func=mdp.feet_at_plane,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "height_offset": 0.035,
        },
    )
    feet_close_xy = RewTerm(
        func=mdp.feet_close_xy_gauss,
        weight=0.4,
        params={
            "threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "std": math.sqrt(0.05),
        },
    )

    # 单个合并奖励 (方案B): 外部 weight=总权重, 内部 proximity/bezier 权重区分状态。
    #   状态切换 (只改内部权重):
    #     proximity_weight=1.0, bezier_weight=0.0 → 仅原始一次性触地奖励
    #     proximity_weight=0.0, bezier_weight=1.0 → 仅 Bézier 连续摆腿跟踪
    #     proximity_weight=1.0, bezier_weight=1.0 → 两者叠加 (默认)
    # 两个分量共享同一 DCM 规划器/相位机/落点缓存 (单实例, 无重复计算)。
    foothold = RewTerm(
        func=mdp.FootholdReward,
        weight=10.0,  # 总奖励权重 (课程 foothold_weight 会 ramp 这个值)
        params={
            # 内部状态权重
            "proximity_weight": 1.0,
            "bezier_weight": 1.0,
            # DCM 规划器落点搜索范围 (沿 x 轴, 前后对称可配)
            #   前方 max_fwd_range 合理区间: [0.4, 0.75]
            #     0.4 = 仅够 0.8 m/s 步态, 无裕量; 0.6 = 覆盖 1.0 m/s 大步 + 上坡/楼梯裕量
            #     (bezier 分支验证值); 网格上限 ±0.9m (37 列 × 0.05m), 超过 0.75 收益递减,
            #     且有效单元线性增加 → 每帧 costmap 算力上升。
            #   后方 max_bwd_range: 0 = 禁用向后落点; 后退/原地步态设 0.1~0.2。
            "max_fwd_range": 0.6,
            "max_bwd_range": 0.0,
            "sigma_p": 10.0,
            # Bézier 参数 (bezier_weight>0 时生效; 默认值与 __init__ 一致)
            "sigma_d": 0.0,
            "T_swing": 0.45,
            "kappa": 0.4,
            "b_min": 0.25,
            "b_max": 0.75,
            "c_min": 0.02,
            "c_scale": 0.2,
            "c_max": 0.1,
            "delta_l_minus": 0.30,
            "delta_l_plus": 0.05,
            "delta_r_minus": 0.05,
            "delta_r_plus": 0.25,
            "heightmap_sensor_cfg": SceneEntityCfg("heightmap"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "foot_center_offset": (0.035, 0.0, -0.058),
            # 接触边沿去抖 (替代 EMA): 连续离地/接触 ≥ edge_hold_time 才触发事件,
            #   且 < edge_window 保证单次触发 (延迟 2-3 控制步, 免疫 <25ms 毛刺)
            "edge_hold_time": 0.025,
            "edge_window": 0.05,
            "debug_vis": False,
            "terrain_names": [
                "pyramid_stairs",
                "pyramid_stairs_inv",
                "up_down",
                "down_up",
            ],
        },
    )

    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "normalize_by_stiffness": True,
        },
    )
    freeze_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.004,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"],
            ),
        },
    )

    # Safety rewards
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={
            "soft_ratio": 0.9,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )


@configclass
class RewardsCfg(MultiRewardCfg):
    rewards: G1Rewards = G1Rewards()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_bound = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5})
    dataset_exhausted = DoneTerm(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
            "reset_without_notice": True,
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    register_virtual_obstacles = EventTerm(
        func=instinct_mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        },
    )

    camera_offsets = EventTerm(
        func=mdp.randomize_camera_offsets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("camera"),
            "offset_pose_ranges": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-0.08, 0.08),
                "pitch": (-0.174, 0.174),
                "yaw": (-0.05, 0.05),
            },
            "distribution": "gaussian",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_without_stand,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "velocity_range": VELOCITY_RANGE,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.tracking_exp_vel,
        params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.0, 0.0)},
    )
    foothold_weight = CurrTerm(
        func=mdp.foothold_weight_schedule,
        params={
            "reward_term_name": "foothold",
            "start_weight": 0.0,
            "end_weight": 40.0,
            "vel_tracking_threshold": 0.75,
            "vel_tracking_target": 0.85,
            # 闩锁阈值:EMA 追踪分 >= 该值后权重与速度解耦(一次性闩锁),
            # 由爬升速度(ramp_rate/ramp_steps)自行单调爬升到 end_weight,不再随追踪值回落。
            # 默认 None = vel_tracking_target。想让追踪到 X 之后"自己慢慢变大",
            # 把这里设成 X,并让 vel_tracking_target 高于 X——否则门控在 X 处
            # 已饱和(权重 == end_weight),闩锁后没有爬升空间只会保持定值。
            "latch_threshold": 0.75,
            # 爬升速度二选一:
            #   ramp_rate  = 每个 env step 叠加的权重增量
            #   ramp_steps = 闩锁后多少 env step 爬满(end_weight)(更直观, 优先级更高)
            # 本环境: 1 env step = 0.02s(50步/秒), 1 episode = 1000 步,
            # 1 轮(PPO iteration) = 24 env step (num_steps_per_env)。
            # 想 2000 轮爬满 -> 2000*24 = 48000 步 -> "ramp_steps": 48000。
            "ramp_steps": 48000,
        },
    )


@configclass
class MonitorCfg:
    pass


##
# Environment configuration
##


@configclass
class ParkourEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**30
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
