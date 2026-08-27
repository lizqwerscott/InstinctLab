from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import omni.kit.app
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_apply_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from instinctlab.tasks.parkour.mdp.dcm_planner import DCMFootholdPlanner
from instinctlab.tasks.parkour.mdp.dcm_visualizer import (
    DCMCostVisualizer,
    clear_markers,
)


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return (
        (dof_error - offset)
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
    )


def feet_close_xy_gauss(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1
    )
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(
            left_height.unsqueeze(-1) - left_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(
            right_height.unsqueeze(-1) - right_sensor_data - height_offset,
            min=0.0,
            max=0.3,
        )
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


class FootholdReward(ManagerTermBase):
    """Phase-driven foothold reward with DCM targets confirmed by real contact.

    A fixed alternating gait phase defines the expected swing trajectory and
    contact schedule. DCM footholds are planned only when the contralateral
    expected stance foot is physically in contact. A real touchdown then
    settles the one-shot foothold accuracy reward and ends that swing task.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._sensor_cfg = cfg.params["sensor_cfg"]
        self._heightmap_sensor_cfg = cfg.params["heightmap_sensor_cfg"]

        # Ankle-roll link origin to the foot-center reference point in foot-local coordinates.
        self._foot_center_offset = torch.tensor(
            cfg.params.get("foot_center_offset", (0.035, 0.0, -0.058)),
            device=env.device,
        )

        # ---- Bézier swing trajectory params (paper Table 1) ----
        self._T_swing = cfg.params.get("T_swing", 0.45)
        self._kappa = cfg.params.get("kappa", 0.4)
        self._b_min = cfg.params.get("b_min", 0.25)
        self._b_max = cfg.params.get("b_max", 0.75)
        self._c_min = cfg.params.get("c_min", 0.05)
        self._c_scale = cfg.params.get("c_scale", 0.5)
        self._c_max = cfg.params.get("c_max", 0.20)
        self._delta_l_minus = cfg.params.get("delta_l_minus", 0.30)
        self._delta_l_plus = cfg.params.get("delta_l_plus", 0.05)
        self._delta_r_minus = cfg.params.get("delta_r_minus", 0.05)
        self._delta_r_plus = cfg.params.get("delta_r_plus", 0.25)
        self._sigma_d = cfg.params.get("sigma_d", 0.0)
        self._sigma_bezier = cfg.params.get("sigma_bezier", 50.0)

        # ---- Alternating gait phase -------------------------------------
        self._duty_factor = cfg.params.get("duty_factor", 0.5)
        self._phase_transition_sigma = cfg.params.get("phase_transition_sigma", 0.04)
        self._phase_speed_threshold = cfg.params.get("phase_speed_threshold", 0.05)
        self._warmup_time_range = cfg.params.get("warmup_time_range", (0.05, 0.15))
        self._swing_contact_weight = cfg.params.get("swing_contact_weight", 0.2)
        if not 0.0 < self._duty_factor < 1.0:
            raise ValueError(f"duty_factor must be in (0, 1), got {self._duty_factor}.")
        if self._phase_transition_sigma <= 0.0:
            raise ValueError(f"phase_transition_sigma must be positive, got {self._phase_transition_sigma}.")
        if self._T_swing <= 0.0:
            raise ValueError(f"T_swing must be positive, got {self._T_swing}.")

        self._gait_frequency = (1.0 - self._duty_factor) / self._T_swing
        self._phase_offset = 0.5
        self._gait_phase = torch.zeros(env.num_envs, device=env.device)
        self._phase_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._walking_last = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._next_left_swing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._warmup_remaining = torch.empty(env.num_envs, device=env.device).uniform_(
            self._warmup_time_range[0], self._warmup_time_range[1]
        )

        # ---- Bézier swing trajectory state ----
        # Lift-off foot centre position (P0) — cached at swing onset
        self._lift_off_pos = torch.zeros(env.num_envs, 2, 3, device=env.device)
        # Bézier apex position (P1) — computed at swing onset
        self._apex_cache = torch.zeros(env.num_envs, 2, 3, device=env.device)
        # u_peak (Eq 10) — pre-computed per foot at swing onset
        self._u_peak_cache = torch.zeros(env.num_envs, 2, device=env.device)
        # ---- Internal state weights (方案B: 一个奖励内部区分状态) ----
        self._proximity_weight = cfg.params.get("proximity_weight", 1.0)
        self._bezier_weight = cfg.params.get("bezier_weight", 1.0)

        # Planner
        self._planner = DCMFootholdPlanner(
            num_envs=env.num_envs,
            device=env.device,
            T=self._T_swing,
            max_fwd_range=cfg.params.get("max_fwd_range", 0.4),
            max_bwd_range=cfg.params.get("max_bwd_range", 0.0),
        )
        masses = env.scene[self._asset_cfg.name].root_physx_view.get_masses()
        self._robot_weight = masses.sum(dim=-1).to(device=env.device) * 9.81
        if self._robot_weight.numel() == 1:
            self._robot_weight = self._robot_weight.expand(env.num_envs)
        # Per-foot caching: [left, right] order (as returned by body_ids)
        # Lazily initialised with actual foot positions on first __call__ to
        # avoid rendering spheres at the world origin (0,0,0).
        self._p_star_cache = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self._p_star_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # True when the left leg is the phase-scheduled swing leg. This is
        # retained for debug visualisation and is derived from _gait_phase.
        self._phase_left_swing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        # ---- Contact edge detection (window gating, replaces EMA) --------
        # 边沿去抖参数 (秒): 连续离地/接触 ≥ edge_hold_time 才视为真实事件,
        #   且 < edge_window 保证每个事件只触发一次 (无状态, 详见 _compute_rewards)。
        self._edge_hold_time = cfg.params.get("edge_hold_time", 0.025)
        self._edge_window = cfg.params.get("edge_window", 0.05)

        # Flag per foot indicating p_star_cache was set by a real plan (not lazy-init).
        # Set True at swing onset, reset False when the swing phase ends.
        self._swing_planned = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

        # Resolve foot order by name (for correct left/right assignment)
        foot_names: list[str] = env.scene[self._asset_cfg.name].data.body_names
        self._foot_order: list[str] = [foot_names[i] for i in self._asset_cfg.body_ids]

        # ---- Optional terrain-specific gating --------------------------------
        self._terrain_names = cfg.params.get("terrain_names", None)
        self._terrain_mask = None

        # ---- Debug visualisation ----
        self._debug_vis = cfg.params.get("debug_vis", False)
        self._debug_vis_handle = None
        self._foothold_visualizer = None
        self._cost_visualizer = None
        self._channels_left = None
        self._channels_right = None
        if self._debug_vis:
            # --- Foothold target spheres ---
            vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/FootholdTargets",
                markers={
                    "left": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                    "right": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._foothold_visualizer = VisualizationMarkers(vis_cfg)
            # Explicitly set visible (defensive — default should be True, but make sure)
            self._foothold_visualizer.set_visibility(True)

            # --- DCM cost-channel heatmap ---
            self._cost_visualizer = DCMCostVisualizer(
                planner=self._planner,
                num_envs=env.num_envs,
                device=env.device,
                active_channel="J",
            )

            app_interface = omni.kit.app.get_app_interface()
            self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
            )

        # ---- Debug: print status ----
        print(
            f"[{self.__class__.__name__}] debug_vis={self._debug_vis}, "
            f"visualizer={'created' if self._foothold_visualizer is not None else 'None'},"
            f" num_envs={env.num_envs}, device={env.device}"
        )

        # ---- 3D visualisation: contact discs, foot->target lines, event flashes ----
        if self._debug_vis:
            # --- Contact state discs (red = stance, blue = swing) ---
            contact_vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/FootholdContactDiscs",
                markers={
                    "contact": sim_utils.CylinderCfg(
                        radius=0.045,
                        height=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "swing": sim_utils.CylinderCfg(
                        radius=0.045,
                        height=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
            self._contact_vis = VisualizationMarkers(contact_vis_cfg)
            self._contact_vis.set_visibility(True)

            # --- Foot->target line markers (white thin cylinders) ---
            line_vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/FootholdTargetLines",
                markers={
                    "line": sim_utils.CylinderCfg(
                        radius=0.004,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                    ),
                },
            )
            self._foot_target_line_vis = VisualizationMarkers(line_vis_cfg)
            self._foot_target_line_vis.set_visibility(True)

            # --- Event flash markers (yellow = swing onset, white = touchdown) ---
            event_vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/FootholdEventFlashes",
                markers={
                    "swing_onset": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
                    ),
                    "touchdown": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                    ),
                },
            )
            self._event_vis = VisualizationMarkers(event_vis_cfg)
            self._event_vis.set_visibility(True)

            # --- Nominal foothold marker (orange sphere) ---
            nominal_vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/NominalFoothold",
                markers={
                    "nominal": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                    ),
                },
            )
            self._nominal_vis = VisualizationMarkers(nominal_vis_cfg)
            self._nominal_vis.set_visibility(True)

            # --- Stair height vertical line (red) ---
            h_line_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/StairHeightLine",
                markers={
                    "line": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._h_line_vis = VisualizationMarkers(h_line_cfg)
            self._h_line_vis.set_visibility(True)

            # --- Event flash timer (4 columns: [L_swing, L_td, R_swing, R_td]) ---
            self._event_timer = torch.zeros(env.num_envs, 4, dtype=torch.int, device=env.device)

        # --- Cached frame data for _debug_vis_callback (always allocated) ---
        self._last_foot_center = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self._last_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self._last_touchdown = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self._last_swing_onset = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self._last_L_nom = None
        self._last_W_nom_left = None
        self._last_W_nom_right = None
        self._last_h_center = None  # (N,) center patch height for stair height line
        self._last_h_fwd = None  # (N,) forward patch height for stair height line

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        # --- Bezier swing curve (dotted spheres along arc) ---
        if self._debug_vis:
            bezier_vis_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/BezierCurve",
                markers={
                    "left": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.6, 1.0),
                        ),
                    ),
                    "right": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.6, 0.0),
                        ),
                    ),
                },
            )
            self._bezier_vis = VisualizationMarkers(bezier_vis_cfg)
            self._bezier_vis.set_visibility(True)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _homogenized_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Map raw phase φ to φ̄ ∈ [0, 1] (HugWBC Eq. 6)."""
        duty = self._duty_factor
        return torch.where(
            phase < duty,
            0.5 * phase / duty,
            0.5 + 0.5 * (phase - duty) / (1.0 - duty),
        )

    def _expected_contact_prob(self, phase: torch.Tensor) -> torch.Tensor:
        """Smooth expected contact probability C(φ) (HugWBC Eq. 5)."""
        phi_bar = self._homogenized_phase(phase)
        sigma = self._phase_transition_sigma
        Phi = self._normal_cdf
        return Phi(phi_bar / sigma) * (1.0 - Phi((phi_bar - 0.5) / sigma)) + Phi((phi_bar - 1.0) / sigma) * (
            1.0 - Phi((phi_bar - 1.5) / sigma)
        )

    def _crossed_into_swing(self, prev_phase: torch.Tensor, curr_phase: torch.Tensor) -> torch.Tensor:
        return (prev_phase < self._duty_factor) & (curr_phase >= self._duty_factor)

    def _swing_progress(self, phase: torch.Tensor) -> torch.Tensor:
        """Normalised swing progress s ∈ [0, 1] inside the swing half-cycle."""
        return ((phase - self._duty_factor) / (1.0 - self._duty_factor)).clamp(0.0, 1.0)

    def _sample_warmup(self, n: int, device: torch.device) -> torch.Tensor:
        lo, hi = self._warmup_time_range
        return torch.empty(n, device=device).uniform_(lo, hi)

    def _cache_bezier_geometry(self, mask: torch.Tensor, foot_idx: int, foot_center: torch.Tensor) -> None:
        """Cache P0 / P1 / u_peak for a planned swing foot."""
        self._lift_off_pos[mask, foot_idx] = foot_center[mask, foot_idx]
        p_l = self._lift_off_pos[mask, foot_idx]
        p_f = self._p_star_cache[mask, foot_idx]
        dz = p_f[:, 2] - p_l[:, 2]
        dz_abs = dz.abs()
        bias = (0.5 + self._kappa * dz / self._planner.h_max).clamp(self._b_min, self._b_max)
        apex_xy = (1.0 - bias).unsqueeze(-1) * p_l[:, :2] + bias.unsqueeze(-1) * p_f[:, :2]
        c = (self._c_min + self._c_scale * dz_abs).clamp(max=self._c_max)
        max_z = torch.max(p_l[:, 2], p_f[:, 2])
        apex_z = 2.0 * (max_z + c) - 0.5 * (p_l[:, 2] + p_f[:, 2])
        self._apex_cache[mask, foot_idx] = torch.stack([apex_xy[:, 0], apex_xy[:, 1], apex_z], dim=-1)

        z_l = p_l[:, 2]
        z_f = p_f[:, 2]
        denom = z_l - 2.0 * apex_z + z_f
        u_peak = torch.where(denom.abs() > 1e-8, (z_l - apex_z) / denom, 0.5 * torch.ones_like(z_l))
        self._u_peak_cache[mask, foot_idx] = u_peak

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        proximity_weight: float | None = None,
        bezier_weight: float | None = None,
        max_fwd_range: float | None = None,
        max_bwd_range: float | None = None,
        sigma_p: float = 10.0,
        sigma_bezier: float | None = None,
        debug_vis: bool = False,
        sigma_d: float | None = None,
        T_swing: float | None = None,
        duty_factor: float | None = None,
        phase_transition_sigma: float | None = None,
        phase_speed_threshold: float | None = None,
        warmup_time_range: tuple[float, float] | None = None,
        swing_contact_weight: float | None = None,
        kappa: float | None = None,
        b_min: float | None = None,
        b_max: float | None = None,
        c_min: float | None = None,
        c_scale: float | None = None,
        c_max: float | None = None,
        delta_l_minus: float | None = None,
        delta_l_plus: float | None = None,
        delta_r_minus: float | None = None,
        delta_r_plus: float | None = None,
        heightmap_sensor_cfg: SceneEntityCfg | None = None,
        asset_cfg: SceneEntityCfg | None = None,
        sensor_cfg: SceneEntityCfg | None = None,
        foot_center_offset: tuple[float, float, float] | None = None,
        edge_hold_time: float | None = None,
        edge_window: float | None = None,
        terrain_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute the phase-driven foothold reward.

        The signature mirrors the config ``params`` keys: Isaac Lab's
        ``_resolve_common_term_cfg`` requires every config param to appear as
        a (defaulted) argument of ``__call__`` or it raises ValueError at
        startup. All values except ``sigma_p`` are resolved in ``__init__``
        from ``cfg.params`` and are intentionally unused here; ``sigma_p`` is
        read directly from the call argument.
        """
        asset = env.scene[self._asset_cfg.name]
        sigma_b = self._sigma_bezier if sigma_bezier is None else sigma_bezier

        # ---- 1. Foot positions & contact --------------------------------
        ankle_pos = asset.data.body_pos_w[:, self._asset_cfg.body_ids]  # (N, 2, 3)
        body_quat = asset.data.body_quat_w[:, self._asset_cfg.body_ids]  # (N, 2, 4)
        foot_center_offset_v = self._foot_center_offset.to(dtype=body_quat.dtype)
        offset_w = quat_apply(
            body_quat.reshape(-1, 4),
            foot_center_offset_v.unsqueeze(0).expand(body_quat.shape[0] * body_quat.shape[1], -1),
        ).reshape(-1, 2, 3)
        foot_center = ankle_pos + offset_w  # (N, 2, 3)

        contact_sensor: ContactSensor = env.scene.sensors[self._sensor_cfg.name]
        net_force = contact_sensor.data.net_forces_w_history  # (N, hist, n_bodies_all)
        contact_norm = torch.norm(net_force[:, -1, self._sensor_cfg.body_ids], dim=-1)  # (N, 2)
        in_contact = contact_norm > 1.0  # (N, 2)

        contact_time = contact_sensor.data.current_contact_time[:, self._sensor_cfg.body_ids]  # (N, 2)
        touchdown = (contact_time > self._edge_hold_time) & (contact_time < self._edge_window)  # (N, 2)

        # ---- 2. Common quantities ---------------------------------------
        root_pos = asset.data.root_pos_w  # (N, 3)
        root_quat = asset.data.root_quat_w  # (N, 4)
        vel_cmd_full = env.command_manager.get_command("base_velocity")  # (N, 3)
        v_cmd_yaw_local = vel_cmd_full[:, :2]
        cmd_speed = torch.norm(v_cmd_yaw_local, dim=1)
        is_walking = cmd_speed > self._phase_speed_threshold

        heightmap = self._get_heightmap(env, root_pos)  # (N, 25, 37)
        com_pos_w = root_pos
        com_vel_w = asset.data.root_lin_vel_w

        newly_uninitialized = ~self._p_star_initialized
        if newly_uninitialized.any():
            self._p_star_cache[newly_uninitialized] = foot_center[newly_uninitialized]
            self._p_star_initialized[newly_uninitialized] = True

        fwd_dist = 0.2
        h_safe = torch.where(torch.isnan(heightmap), torch.zeros_like(heightmap), heightmap)
        row_c, col_c = heightmap.shape[1] // 2, heightmap.shape[2] // 2
        col_f = col_c + int(round(fwd_dist / self._planner.cell_size))
        col_f = min(max(col_f, 1), heightmap.shape[2] - 2)
        center_patch = h_safe[:, row_c - 1 : row_c + 2, col_c - 1 : col_c + 2]
        fwd_patch = h_safe[:, row_c - 1 : row_c + 2, col_f - 1 : col_f + 2]
        h_center = center_patch.mean(dim=(1, 2))
        h_fwd = fwd_patch.mean(dim=(1, 2))
        k = (h_fwd - h_center) / self._planner.T
        if self._debug_vis:
            self._last_h_center = h_center.detach().clone()
            self._last_h_fwd = h_fwd.detach().clone()
            self._last_fwd_dist = fwd_dist

        # ---- 3. Phase clock (open-loop, walking only) -------------------
        became_walking = is_walking & ~self._walking_last
        became_standing = (~is_walking) & self._walking_last

        if became_walking.any():
            n_bw = int(became_walking.sum().item())
            self._warmup_remaining[became_walking] = self._sample_warmup(n_bw, env.device)
            self._phase_active[became_walking] = False
            self._swing_planned[became_walking] = False

        if became_standing.any():
            self._phase_active[became_standing] = False
            self._swing_planned[became_standing] = False
            self._warmup_remaining[became_standing] = 0.0

        warming = is_walking & ~self._phase_active
        if warming.any():
            self._warmup_remaining[warming] = self._warmup_remaining[warming] - env.step_dt
        warmup_done = warming & (self._warmup_remaining <= 0.0)

        prev_phase = self._gait_phase.clone()
        advance_mask = is_walking & self._phase_active
        if advance_mask.any():
            self._gait_phase[advance_mask] = (
                self._gait_phase[advance_mask] + self._gait_frequency * env.step_dt
            ) % 1.0

        start_left = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if warmup_done.any():
            n_wd = int(warmup_done.sum().item())
            start_left[warmup_done] = torch.rand(n_wd, device=env.device) > 0.5
            self._next_left_swing[warmup_done] = start_left[warmup_done]
            # Left swing onset: φ_L = duty; right swing onset: φ_L = 0 → φ_R = duty.
            self._gait_phase[warmup_done] = torch.where(
                start_left[warmup_done],
                torch.full((n_wd,), self._duty_factor, device=env.device),
                torch.zeros(n_wd, device=env.device),
            )
            self._phase_active[warmup_done] = True
            self._warmup_remaining[warmup_done] = 0.0

        self._walking_last = is_walking

        phase_L = self._gait_phase
        phase_R = (self._gait_phase + self._phase_offset) % 1.0
        phases = torch.stack([phase_L, phase_R], dim=1)  # (N, 2)
        prev_phases = torch.stack(
            [prev_phase, (prev_phase + self._phase_offset) % 1.0],
            dim=1,
        )
        contact_prob = self._expected_contact_prob(phases)  # (N, 2)
        swing_weight = 1.0 - contact_prob  # (N, 2)

        phase_onset = torch.stack(
            [
                self._crossed_into_swing(prev_phases[:, 0], phases[:, 0]),
                self._crossed_into_swing(prev_phases[:, 1], phases[:, 1]),
            ],
            dim=1,
        )
        if warmup_done.any():
            phase_onset[warmup_done, 0] = phase_onset[warmup_done, 0] | start_left[warmup_done]
            phase_onset[warmup_done, 1] = phase_onset[warmup_done, 1] | (~start_left[warmup_done])

        # Only active walking envs produce scheduling edges.
        phase_onset = phase_onset & self._phase_active.unsqueeze(-1)
        self._phase_left_swing = phases[:, 0] >= self._duty_factor

        # ---- 4. Plan DCM targets at phase swing-onset -------------------
        for foot_idx in range(2):
            onset_mask = phase_onset[:, foot_idx]
            if not onset_mask.any():
                continue
            stance_idx = 1 - foot_idx
            can_plan = onset_mask & in_contact[:, stance_idx]
            if not can_plan.any():
                # No contralateral stance contact → skip DCM for this onset.
                self._swing_planned[onset_mask, foot_idx] = False
                continue

            swing_sign = 1.0 if foot_idx == 0 else -1.0
            p_new = self._planner.plan_in_world(
                heightmap[can_plan],
                v_cmd_yaw_local[can_plan],
                foot_center[can_plan, stance_idx],
                root_pos[can_plan],
                root_quat[can_plan],
                torch.full((int(can_plan.sum().item()),), swing_sign, device=env.device),
                com_pos_w=com_pos_w[can_plan],
                com_vel_w=com_vel_w[can_plan],
                k=k[can_plan],
            )
            self._p_star_cache[can_plan, foot_idx] = p_new
            self._swing_planned[can_plan, foot_idx] = True
            # Onsets that fail can_plan stay unplanned.
            failed = onset_mask & ~can_plan
            if failed.any():
                self._swing_planned[failed, foot_idx] = False
            self._cache_bezier_geometry(can_plan, foot_idx, foot_center)

        # -- Cost-channel visualisation (full planning each frame) --
        if self._debug_vis and self._cost_visualizer is not None:
            _, self._channels_left = self._planner.plan_with_channels_in_world(
                heightmap,
                v_cmd_yaw_local,
                foot_center[:, 1],
                root_pos,
                root_quat,
                torch.ones(env.num_envs, device=env.device),
                com_pos_w=com_pos_w,
                com_vel_w=com_vel_w,
                k=k,
            )
            _, self._channels_right = self._planner.plan_with_channels_in_world(
                heightmap,
                v_cmd_yaw_local,
                foot_center[:, 0],
                root_pos,
                root_quat,
                -torch.ones(env.num_envs, device=env.device),
                com_pos_w=com_pos_w,
                com_vel_w=com_vel_w,
                k=k,
            )
            self._last_heightmap = heightmap
            self._last_root_pos = root_pos
            self._last_root_quat = root_quat
            self._last_L_nom = self._channels_left["L_nom"]
            self._last_W_nom_left = self._channels_left["W_nom"]
            self._last_W_nom_right = self._channels_right["W_nom"]

        # ---- 5. Rewards -------------------------------------------------
        touchdown_active = touchdown & self._swing_planned  # (N, 2)
        dense_scale = env.step_dt / self._T_swing
        reward = torch.zeros(env.num_envs, device=env.device)
        active_env = self._phase_active  # walking + past warm-up

        for foot_idx in range(2):
            # Continuous swing-contact force penalty (phase-weighted).
            force_ratio = (contact_norm[:, foot_idx] / self._robot_weight).clamp(0.0, 1.0)
            reward = reward - (
                dense_scale
                * self._swing_contact_weight
                * swing_weight[:, foot_idx]
                * (force_ratio**2)
                * active_env.float()
            )

            # Dense Bézier tracking while a plan is active (stops on touchdown).
            tracking_mask = self._swing_planned[:, foot_idx] & active_env
            if self._bezier_weight > 0.0 and tracking_mask.any():
                s = self._swing_progress(phases[tracking_mask, foot_idx])
                u = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
                u_unsq = u.unsqueeze(-1)

                P0 = self._lift_off_pos[tracking_mask, foot_idx]
                P1 = self._apex_cache[tracking_mask, foot_idx]
                P2 = self._p_star_cache[tracking_mask, foot_idx]
                one_minus_u = 1.0 - u_unsq
                p_bezier = one_minus_u**2 * P0 + 2.0 * one_minus_u * u_unsq * P1 + u_unsq**2 * P2

                pos_err = foot_center[tracking_mask, foot_idx] - p_bezier
                pos_err_sq = (pos_err**2).sum(dim=-1)

                if self._sigma_d > 0.0:
                    p_dot = 2.0 * one_minus_u * (P1 - P0) + 2.0 * u_unsq * (P2 - P1)
                    p_dot_norm = torch.norm(p_dot, dim=-1, keepdim=True).clamp(min=1e-8)
                    p_dot_unit = p_dot / p_dot_norm
                    u_peak_val = self._u_peak_cache[tracking_mask, foot_idx]
                    pre_mask = (u >= u_peak_val - self._delta_l_minus) & (u < u_peak_val - self._delta_l_plus)
                    t_hat_pre = torch.stack([p_dot_unit[:, 2], p_dot_unit[:, 1], -p_dot_unit[:, 0]], dim=-1)
                    t_hat_pre = t_hat_pre / torch.norm(t_hat_pre, dim=-1, keepdim=True).clamp(min=1e-8)
                    t_hat = torch.where(pre_mask.unsqueeze(-1), t_hat_pre, torch.zeros_like(p_dot_unit))
                    post_mask = (u > u_peak_val + self._delta_r_minus) & (u <= u_peak_val + self._delta_r_plus)
                    t_hat = torch.where(post_mask.unsqueeze(-1), p_dot_unit, t_hat)

                    foot_quat = body_quat[tracking_mask, foot_idx]
                    e_x = torch.tensor([1.0, 0.0, 0.0], device=foot_quat.device, dtype=foot_quat.dtype)
                    e_x = e_x.unsqueeze(0).expand(int(tracking_mask.sum().item()), -1)
                    d_hat_f = quat_apply(foot_quat, e_x)
                    ori_active = (pre_mask | post_mask).float()
                    ori_err_sq = ((d_hat_f - t_hat) ** 2).sum(dim=-1) * ori_active
                else:
                    ori_err_sq = 0.0

                tracking_quality = torch.exp(-sigma_b * pos_err_sq - self._sigma_d * ori_err_sq)
                bounded = 2.0 * tracking_quality - 1.0
                reward[tracking_mask] = reward[tracking_mask] + (
                    dense_scale
                    * self._bezier_weight
                    * swing_weight[tracking_mask, foot_idx]
                    * u
                    * bounded
                )

            # One-shot foothold accuracy at real touchdown, gated by C(φ).
            td_mask = touchdown_active[:, foot_idx] & active_env
            if self._proximity_weight > 0.0 and td_mask.any():
                dist = foot_center[td_mask, foot_idx] - self._p_star_cache[td_mask, foot_idx]
                dist_sq = (dist**2).sum(dim=-1)
                foot_reward = contact_prob[td_mask, foot_idx] * torch.exp(-sigma_p * dist_sq)
                reward[td_mask] = reward[td_mask] + self._proximity_weight * foot_reward

            if td_mask.any():
                self._swing_planned[td_mask, foot_idx] = False

        if self._terrain_names is not None:
            self._update_terrain_mask(env)
            if self._terrain_mask is not None:
                reward = reward * self._terrain_mask.float()

        # ---- 6. Debug frame cache ---------------------------------------
        self._last_foot_center = foot_center.clone()
        self._last_contact = in_contact.clone()
        self._last_touchdown = touchdown_active.clone()
        self._last_swing_onset = phase_onset.clone()

        if self._debug_vis and hasattr(self, "_event_timer"):
            self._event_timer[phase_onset[:, 0], 0] = 10
            self._event_timer[touchdown_active[:, 0], 1] = 15
            self._event_timer[phase_onset[:, 1], 2] = 10
            self._event_timer[touchdown_active[:, 1], 3] = 15
            self._event_timer = (self._event_timer - 1).clamp(min=0)

        return reward  # (N,)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_from_z_to_dir(dir: torch.Tensor) -> torch.Tensor:
        """Compute (N, 4) quaternion (w, x, y, z) rotating local z-axis to *dir*."""
        z = torch.zeros_like(dir)
        z[:, 2] = 1.0
        dot = (z * dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        cross = torch.cross(z, dir, dim=-1)
        cross_norm = torch.norm(cross, dim=-1, keepdim=True)
        # Parallel case (direction already aligned with z): identity quaternion
        parallel = cross_norm.squeeze(-1) < 1e-8
        # Half-angle trig
        half_angle = torch.acos(dot) * 0.5
        axis = cross / (cross_norm + 1e-8)
        quat = torch.zeros(dir.shape[0], 4, device=dir.device)
        quat[:, 0] = torch.cos(half_angle).squeeze(-1)
        quat[:, 1:] = axis * torch.sin(half_angle)
        quat[parallel, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dir.device)
        return quat  # (N, 4) in (w, x, y, z) format — matches VisualizationMarkers

    # ------------------------------------------------------------------

    def _debug_vis_callback(self, event):
        """Render cached foothold target markers + DCM cost heatmap + 3D overlays every frame."""
        # ---- Foothold spheres ----
        if self._foothold_visualizer is not None:
            n = self._p_star_cache.shape[0]
            if n > 0:
                vis_mask = self._terrain_mask if self._terrain_mask is not None else None
                if vis_mask is not None:
                    p_star = self._p_star_cache[vis_mask]  # (M, 2, 3)
                else:
                    p_star = self._p_star_cache  # (N, 2, 3)
                if p_star.shape[0] > 0:
                    # (M, 2, 3) -> (2, M, 3) -> (2*M, 3)
                    poses = p_star.permute(1, 0, 2).reshape(-1, 3)
                    m = p_star.shape[0]
                    marker_indices = torch.zeros(2 * m, dtype=torch.int, device=p_star.device)
                    marker_indices[m:] = 1  # right foot -> green
                    self._foothold_visualizer.visualize(poses, marker_indices=marker_indices)

        # ---- DCM cost heatmap ----
        if self._cost_visualizer is not None and self._channels_left is not None and self._channels_right is not None:
            # For each environment, pick the channel dict that corresponds to
            # the swinging foot (left-swing  → channels_left,  right-swing → channels_right).
            # The visualiser's update() already skips envs where both feet are
            # in contact, so we pass in_contact along.
            in_contact = self._last_contact  # (N, 2) bool — 最近一帧电平缓存

            # Build a merged channels dict: pick left or right data per-environment.
            merged: dict[str, torch.Tensor] = {}
            for key in self._channels_left:
                left_val = self._channels_left[key]  # (N, H, W) or (N,)
                right_val = self._channels_right[key]  # (N, H, W) or (N,)
                if left_val.dim() == 1:
                    merged[key] = torch.where(
                        in_contact[:, 0],  # (N,)
                        right_val,  # (N,)
                        left_val,  # (N,)
                    )
                else:
                    merged[key] = torch.where(
                        in_contact[:, 0:1, None],  # (N, 1, 1)
                        right_val,
                        left_val,
                    )

            self._cost_visualizer.update(
                channels=merged,
                heightmap=self._last_heightmap,
                root_pos_w=self._last_root_pos,
                root_quat_w=self._last_root_quat,
                in_contact=in_contact,
            )

        # ================================================================
        # 3D Scene overlays (contact discs + foot->target lines + events)
        # ================================================================

        # ---- Contact state discs (slightly below foot, red=stance, blue=swing) ----
        if hasattr(self, "_contact_vis") and self._contact_vis is not None:
            vis_mask = self._terrain_mask if self._terrain_mask is not None else None
            if vis_mask is not None:
                foot_pos = self._last_foot_center[vis_mask]
                last_contact = self._last_contact[vis_mask]
            else:
                foot_pos = self._last_foot_center
                last_contact = self._last_contact
            if foot_pos.shape[0] > 0:
                foot_pos_ground = foot_pos.clone()
                foot_pos_ground[:, :, 2] -= 0.02  # place slightly below foot
                poses_flat = foot_pos_ground.reshape(-1, 3)  # (M*2, 3)
                contact_flat = (~last_contact).reshape(-1).long()  # (M*2,)
                self._contact_vis.visualize(
                    translations=poses_flat,
                    marker_indices=contact_flat,
                )

        # ---- Foot->target lines (white cylinders from foot to p_star_cache) ----
        if hasattr(self, "_foot_target_line_vis") and self._foot_target_line_vis is not None:
            vis_mask = self._terrain_mask if self._terrain_mask is not None else None
            if vis_mask is not None:
                p_star = self._p_star_cache[vis_mask]  # (M, 2, 3)
                foot_pos = self._last_foot_center[vis_mask]  # (M, 2, 3)
            else:
                p_star = self._p_star_cache
                foot_pos = self._last_foot_center
            if foot_pos.shape[0] > 0:
                dirs = p_star - foot_pos  # (M, 2, 3)
                lengths = torch.norm(dirs, dim=-1)  # (M, 2)
                valid_mask = lengths > 0.01
                valid_flat = valid_mask.reshape(-1)  # (M*2,)
                if valid_flat.any():
                    foot_flat = foot_pos.reshape(-1, 3)[valid_flat]
                    target_flat = p_star.reshape(-1, 3)[valid_flat]
                    dir_flat = dirs.reshape(-1, 3)[valid_flat]
                    len_flat = lengths.reshape(-1)[valid_flat]
                    mid_flat = (foot_flat + target_flat) * 0.5
                    orient_flat = self._quat_from_z_to_dir(dir_flat)
                    scales_flat = torch.ones_like(mid_flat)
                    scales_flat[:, 2] = len_flat
                    self._foot_target_line_vis.visualize(
                        translations=mid_flat,
                        orientations=orient_flat,
                        scales=scales_flat,
                        marker_indices=torch.zeros(valid_flat.sum(), dtype=torch.int, device=dir_flat.device),
                    )
                else:
                    clear_markers(self._foot_target_line_vis, self._planner.device)

        # ---- Event flash markers (yellow=swing_onset, white=touchdown) ----
        if hasattr(self, "_event_vis") and self._event_vis is not None:
            vis_mask = self._terrain_mask if self._terrain_mask is not None else None
            if vis_mask is not None:
                event_timer = self._event_timer[vis_mask]
                foot_pos = self._last_foot_center[vis_mask]
            else:
                event_timer = self._event_timer
                foot_pos = self._last_foot_center
            if foot_pos.shape[0] > 0:
                event_positions = []
                event_indices = []
                for foot_idx in range(2):
                    # Swing onset timer columns: 0 for left, 2 for right
                    swing_timer = event_timer[:, foot_idx * 2]
                    swing_active = swing_timer > 0
                    if swing_active.any():
                        event_positions.append(foot_pos[swing_active, foot_idx])
                        event_indices.append(
                            torch.zeros(
                                swing_active.sum(),
                                dtype=torch.int,
                                device=swing_timer.device,
                            )
                        )
                    # Touchdown timer columns: 1 for left, 3 for right
                    td_timer = event_timer[:, foot_idx * 2 + 1]
                    td_active = td_timer > 0
                    if td_active.any():
                        event_positions.append(foot_pos[td_active, foot_idx])
                        event_indices.append(torch.ones(td_active.sum(), dtype=torch.int, device=td_timer.device))
                if event_positions:
                    all_pos = torch.cat(event_positions, dim=0)
                    all_idx = torch.cat(event_indices, dim=0)
                    self._event_vis.visualize(
                        translations=all_pos,
                        marker_indices=all_idx,
                    )
                else:
                    clear_markers(self._event_vis, self._planner.device)

        # ---- Nominal foothold marker (orange sphere at L_nom, W_nom, terrain_z) ----
        if hasattr(self, "_nominal_vis") and self._nominal_vis is not None:
            if (
                self._last_L_nom is not None
                and self._last_W_nom_left is not None
                and self._last_W_nom_right is not None
            ):
                vis_mask = self._terrain_mask if self._terrain_mask is not None else None
                L_nom_flat = self._last_L_nom.squeeze(-1).squeeze(-1)  # (N,)
                # Pick W_nom based on actual swing foot
                W_nom_flat = torch.where(
                    self._phase_left_swing,
                    self._last_W_nom_left.squeeze(-1).squeeze(-1),  # (N,)  left swing  → +lp
                    self._last_W_nom_right.squeeze(-1).squeeze(-1),  # (N,)  right swing → -lp
                )

                # Grid indices for terrain height lookup
                cell_size = self._planner.cell_size
                g_w, g_h = self._planner.grid_w, self._planner.grid_h
                ix = (L_nom_flat / cell_size + (g_w - 1) / 2).round().long().clamp(0, g_w - 1)
                iy = (W_nom_flat / cell_size + (g_h - 1) / 2).round().long().clamp(0, g_h - 1)

                # Sample terrain height from heightmap (h_safe equivalent via 0-fill)
                h_map = self._last_heightmap  # (N, H, W)
                N = h_map.shape[0]
                terrain_z = torch.where(
                    torch.isnan(h_map[torch.arange(N, device=h_map.device), iy, ix]),
                    torch.zeros(N, device=h_map.device),
                    h_map[torch.arange(N, device=h_map.device), iy, ix],
                )  # (N,)

                # Nominal foothold in pelvis-local frame
                nominal_local = torch.stack([L_nom_flat, W_nom_flat, terrain_z], dim=-1)  # (N, 3)

                nominal_world = quat_apply_yaw(self._last_root_quat, nominal_local) + self._last_root_pos  # (N, 3)

                # Apply terrain mask
                if vis_mask is not None:
                    nominal_world = nominal_world[vis_mask]
                if nominal_world.shape[0] > 0:
                    self._nominal_vis.visualize(
                        translations=nominal_world,
                        marker_indices=torch.zeros(
                            nominal_world.shape[0],
                            dtype=torch.int,
                            device=nominal_world.device,
                        ),
                    )
                else:
                    clear_markers(self._nominal_vis, self._planner.device)

        # ---- Stair height vertical line (from h_center to h_fwd at fwd_dist ahead) ----
        if hasattr(self, "_h_line_vis") and self._h_line_vis is not None and self._last_h_center is not None:
            h_c = self._last_h_center  # (N,)
            h_f = self._last_h_fwd  # (N,)
            vis_mask = self._terrain_mask if self._terrain_mask is not None else None
            if vis_mask is not None:
                h_c = h_c[vis_mask]
                h_f = h_f[vis_mask]
            n_valid = h_c.shape[0]
            if n_valid > 0:
                # Forward position in local frame: (fwd_dist, 0, 0)
                fwd_local = torch.zeros(n_valid, 3, device=h_c.device)
                fwd_local[:, 0] = self._last_fwd_dist

                # Rotate to world frame
                if vis_mask is not None:
                    root_pos_v = self._last_root_pos[vis_mask]
                    root_quat_v = self._last_root_quat[vis_mask]
                else:
                    root_pos_v = self._last_root_pos
                    root_quat_v = self._last_root_quat

                fwd_xy_w = quat_apply_yaw(root_quat_v, fwd_local) + root_pos_v

                # Bottom at forward terrain height, top at forward terrain + step_height.
                # This ensures the line sits on the actual terrain at its XY position
                # instead of piercing through the ground when going up stairs.
                bottom_w = fwd_xy_w.clone()
                bottom_w[:, 2] = root_pos_v[:, 2] + h_f
                top_w = fwd_xy_w.clone()
                top_w[:, 2] = root_pos_v[:, 2] + h_f + (h_f - h_c)  # = 2*h_f - h_c

                dir_w = top_w - bottom_w
                len_w = torch.norm(dir_w, dim=-1)
                valid_line = len_w > 0.005
                if valid_line.any():
                    mid_w = (bottom_w + top_w) * 0.5
                    scales = torch.ones(n_valid, 3, device=h_c.device)
                    scales[:, 2] = len_w
                    orient_w = self._quat_from_z_to_dir(dir_w)
                    self._h_line_vis.visualize(
                        translations=mid_w[valid_line],
                        orientations=orient_w[valid_line],
                        scales=scales[valid_line],
                        marker_indices=None,
                    )
                else:
                    clear_markers(self._h_line_vis, self._planner.device)
            else:
                clear_markers(self._h_line_vis, self._planner.device)

        # ---- Bezier swing curve (dotted spheres along arc) ----
        if hasattr(self, "_bezier_vis") and self._bezier_vis is not None:
            vis_mask = self._terrain_mask if self._terrain_mask is not None else None
            K = 15
            bezier_poses = []
            bezier_indices = []

            for foot_idx in range(2):
                if vis_mask is not None:
                    foot_active = vis_mask & self._swing_planned[:, foot_idx]
                else:
                    foot_active = self._swing_planned[:, foot_idx]

                active_ids = foot_active.nonzero(as_tuple=True)[0]
                if active_ids.numel() == 0:
                    continue

                P0 = self._lift_off_pos[active_ids, foot_idx]
                P1 = self._apex_cache[active_ids, foot_idx]
                P2 = self._p_star_cache[active_ids, foot_idx]
                M = active_ids.numel()

                us = torch.linspace(0.0, 1.0, K, device=P0.device)
                u_unsq = us.view(1, -1, 1)
                one_minus_u = 1.0 - u_unsq

                points = (
                    one_minus_u**2 * P0.unsqueeze(1)
                    + 2.0 * one_minus_u * u_unsq * P1.unsqueeze(1)
                    + u_unsq**2 * P2.unsqueeze(1)
                )

                points_flat = points.reshape(-1, 3)
                idx_flat = torch.full(
                    (points_flat.shape[0],),
                    foot_idx,
                    dtype=torch.int,
                    device=points.device,
                )
                bezier_poses.append(points_flat)
                bezier_indices.append(idx_flat)

            if bezier_poses:
                all_poses = torch.cat(bezier_poses, dim=0)
                all_idx = torch.cat(bezier_indices, dim=0)
                self._bezier_vis.visualize(translations=all_poses, marker_indices=all_idx)
            else:
                clear_markers(self._bezier_vis, self._planner.device)

    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset per-env caches (called by RewardManager on env reset)."""
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            n = self._gait_phase.shape[0]
            device = self._gait_phase.device
        else:
            n = env_ids.numel()
            device = self._gait_phase.device

        self._p_star_cache[env_ids] = 0.0
        self._p_star_initialized[env_ids] = False
        self._swing_planned[env_ids] = False
        self._lift_off_pos[env_ids] = 0.0
        self._apex_cache[env_ids] = 0.0
        self._u_peak_cache[env_ids] = 0.0

        self._gait_phase[env_ids] = 0.0
        self._phase_active[env_ids] = False
        self._walking_last[env_ids] = False
        self._next_left_swing[env_ids] = False
        self._warmup_remaining[env_ids] = self._sample_warmup(n, device)
        self._phase_left_swing[env_ids] = True

        if hasattr(self, "_event_timer"):
            self._event_timer[env_ids] = 0
        self._terrain_mask = None
        self._last_L_nom = None
        self._last_W_nom_left = None
        self._last_W_nom_right = None
        self._last_h_center = None
        self._last_h_fwd = None
        self._last_fwd_dist = None

    # ------------------------------------------------------------------

    def _update_terrain_mask(self, env: "ManagerBasedRLEnv"):
        """Compute and cache (N,) bool mask for terrain-name gating.

        Column-to-sub-terrain mapping is derived from the relative proportions
        defined in the terrain-generator config.  The mask is recomputed
        on the first call and after every reset.
        """
        terrain = env.scene["terrain"]
        cfg = terrain.cfg.terrain_generator
        sub_names = list(cfg.sub_terrains.keys())
        proportions = np.array(
            [cfg.sub_terrains[n].proportion for n in sub_names],
            dtype=np.float64,
        )
        proportions /= np.sum(proportions)

        sub_indices = np.empty(cfg.num_cols, dtype=np.int32)
        cumsum = np.cumsum(proportions)
        for col in range(cfg.num_cols):
            sub_indices[col] = int(np.min(np.where(col / cfg.num_cols + 0.001 < cumsum)[0]))

        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in self._terrain_names:
            if name not in sub_names:
                continue
            type_idx = sub_names.index(name)
            for col_idx in np.where(sub_indices == type_idx)[0]:
                env_ids = torch.where(terrain.terrain_types == col_idx)[0]
                mask[env_ids] = True
        self._terrain_mask = mask

    # ------------------------------------------------------------------

    def _get_heightmap(self, env: "ManagerBasedRLEnv", root_pos: torch.Tensor) -> torch.Tensor:
        """Return (N, 25, 37) pelvis-local terrain heights (NaN = ray missed).

        The z-component of (hit_world - root_pos) gives the height
        relative to the pelvis origin.  This is frame-rotation-invariant
        (pure z differencing), so no yaw handling is needed here.
        """
        sensor = env.scene[self._heightmap_sensor_cfg.name]
        hits_w = sensor.data.ray_hits_w  # (N, num_rays, 3)
        num_rays = hits_w.shape[1]
        H, W = 25, num_rays // 25
        # Relative height (world z - pelvis z)
        z_rel = (hits_w[..., 2] - root_pos[:, 2].unsqueeze(1)).view(-1, H, W)
        # Mark ray-miss: hit world-z far below any reasonable terrain
        missed = hits_w.view(-1, H, W, 3)[..., 2] < -100.0
        return torch.where(missed, torch.full_like(z_rel, float("nan")), z_rel)
