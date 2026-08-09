from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import omni.kit.app
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
    """Combined foothold reward: sparse touchdown (proximity) + dense per-frame
    Bézier swing tracking in a single reward term.

    The external RewTerm weight scales the total reward. Inside the term,
    ``proximity_weight`` and ``bezier_weight`` select the state:
      - proximity_weight > 0, bezier_weight == 0  → sparse touchdown reward only
      - proximity_weight == 0, bezier_weight > 0  → dense Bézier tracking only
      - both > 0                                  → both components summed
    The two components share one DCM planner, one phase-state machine and one
    foothold-target cache (single per-frame pipeline). No touchdown/over-time
    penalties are applied.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._sensor_cfg = cfg.params["sensor_cfg"]
        self._heightmap_sensor_cfg = cfg.params["heightmap_sensor_cfg"]

        # 踝关节到脚掌中心的偏移量（沿脚部前向 +x），可配置参数
        self._ankle_offset = cfg.params.get("ankle_offset", 0.035)

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

        # ---- Bézier swing trajectory state ----
        # Swing elapsed timer (seconds since lift-off, per foot)
        self._swing_elapsed = torch.zeros(env.num_envs, 2, device=env.device)
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
            max_fwd_range=cfg.params.get("max_fwd_range", 0.4),
            max_bwd_range=cfg.params.get("max_bwd_range", 0.0),
        )
        # Per-foot caching: [left, right] order (as returned by body_ids)
        # Lazily initialised with actual foot positions on first __call__ to
        # avoid rendering spheres at the world origin (0,0,0).
        self._p_star_cache = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self._p_star_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # ---- Phase-state machine for swing-leg tracking -------------------
        # True  = left  leg is the swing leg (right leg is stance)
        # False = right leg is the swing leg (left  leg is stance)
        # Initialised to left-swing by default; reset() can randomise.
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
        self._last_body_pos = torch.zeros(env.num_envs, 2, 3, device=env.device)
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

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        proximity_weight: float | None = None,
        bezier_weight: float | None = None,
        max_fwd_range: float | None = None,
        max_bwd_range: float | None = None,
        sigma_p: float = 10.0,
        debug_vis: bool = False,
        sigma_d: float | None = None,
        T_swing: float | None = None,
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
        ankle_offset: float | None = None,
        edge_hold_time: float | None = None,
        edge_window: float | None = None,
        terrain_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute the combined foothold reward.

        The signature mirrors the config ``params`` keys: Isaac Lab's
        ``_resolve_common_term_cfg`` requires every config param to appear as
        a (defaulted) argument of ``__call__`` or it raises ValueError at
        startup. All values except ``sigma_p`` are resolved in ``__init__``
        from ``cfg.params`` and are intentionally unused here; ``sigma_p`` is
        read directly from the call argument.
        """
        asset = env.scene[self._asset_cfg.name]

        # ---- 1. Foot positions & contact (raw) --------------------------
        body_pos = asset.data.body_pos_w[:, self._asset_cfg.body_ids]  # (N, 2, 3) — ankle
        body_quat = asset.data.body_quat_w[:, self._asset_cfg.body_ids]  # (N, 2, 4) — foot orientation
        # 将踝关节位置偏移到脚掌中心（沿脚部局部 +x 方向）
        ankle_offset_v = torch.tensor([self._ankle_offset, 0.0, 0.0], device=env.device)
        offset_w = quat_apply(
            body_quat.reshape(-1, 4),
            ankle_offset_v.unsqueeze(0).expand(body_quat.shape[0] * body_quat.shape[1], -1),
        ).reshape(-1, 2, 3)
        foot_center = body_pos + offset_w  # (N, 2, 3) — foot center

        contact_sensor: ContactSensor = env.scene.sensors[self._sensor_cfg.name]
        net_force = contact_sensor.data.net_forces_w_history  # (N, hist, n_bodies_all)
        contact_norm = torch.norm(net_force[:, -1, self._sensor_cfg.body_ids], dim=-1)  # (N, 2)
        in_contact = contact_norm > 1.0  # (N, 2) 电平: 当前子步是否接触

        # ---- 2a. Edge detection: window gating on sensor-integrated times ----
        # 传感器在 200Hz 下累积 current_air_time / current_contact_time
        # (内部用 force_threshold 判接触)。用"最小持续时间"窗口做边沿去抖 (替代 EMA):
        #   - 短毛刺 (< edge_hold_time) 采样到的时间量不够长 → 不触发
        #   - 真实事件延迟 2-3 个控制步 (40-60ms) 触发
        #   - 窗口上限 edge_window 保证每个事件只触发一次, 无需上一帧状态
        air_time = contact_sensor.data.current_air_time[:, self._sensor_cfg.body_ids]  # (N, 2)
        contact_time = contact_sensor.data.current_contact_time[:, self._sensor_cfg.body_ids]  # (N, 2)
        swing_onset = (air_time > self._edge_hold_time) & (air_time < self._edge_window)  # (N, 2)
        touchdown = (contact_time > self._edge_hold_time) & (contact_time < self._edge_window)  # (N, 2)

        # ---- 2c. Phase-state update --------------------------------------
        # When a foot loses contact it becomes the new swing leg.
        #   swing_onset[:, 0] == True  → left foot just lifted  → phase_left=True
        #   swing_onset[:, 1] == True  → right foot just lifted → phase_left=False
        envs_to_swap = swing_onset[:, 0] | swing_onset[:, 1]
        new_phase = swing_onset[:, 0]  # True if left onset, False if right onset
        self._phase_left_swing = torch.where(envs_to_swap, new_phase, self._phase_left_swing)

        # ---- 3. Common quantities (shared by planning & reward) ----------
        root_pos = asset.data.root_pos_w  # (N, 3)
        root_quat = asset.data.root_quat_w  # (N, 4) w,x,y,z
        v_cmd = env.command_manager.get_command("base_velocity")[:, :2]  # (N, 2) world
        heightmap = self._get_heightmap(env, root_pos)  # (N, 25, 37)

        # Approximate CoM state from root (pelvis) state.
        com_pos_w = root_pos  # (N, 3)
        com_vel_w = asset.data.root_lin_vel_w  # (N, 3)

        # ---- 3a. Lazy-init p_star_cache with actual foot positions -------
        newly_uninitialized = ~self._p_star_initialized
        if newly_uninitialized.any():
            self._p_star_cache[newly_uninitialized, 0] = foot_center[newly_uninitialized, 0]
            self._p_star_cache[newly_uninitialized, 1] = foot_center[newly_uninitialized, 1]
            self._p_star_initialized[newly_uninitialized] = True

        # ---- 动态 k：从高度图前方 fwd_dist 与中心的高度差除以 T ----
        fwd_dist = 0.2  # 向前采样距离 (m)
        h_safe = torch.where(torch.isnan(heightmap), torch.zeros_like(heightmap), heightmap)
        row_c, col_c = heightmap.shape[1] // 2, heightmap.shape[2] // 2  # 中心 (y=0, x=0)
        col_f = col_c + int(round(fwd_dist / self._planner.cell_size))  # 前方 fwd_dist
        col_f = min(max(col_f, 1), heightmap.shape[2] - 2)  # clamp 防越界
        center_patch = h_safe[:, row_c - 1 : row_c + 2, col_c - 1 : col_c + 2]  # (N, 3, 3)
        fwd_patch = h_safe[:, row_c - 1 : row_c + 2, col_f - 1 : col_f + 2]  # (N, 3, 3)
        h_center = center_patch.mean(dim=(1, 2))  # (N,)
        h_fwd = fwd_patch.mean(dim=(1, 2))  # (N,)
        k = (h_fwd - h_center) / self._planner.T
        if self._debug_vis:
            self._last_h_center = h_center.detach().clone()
            self._last_h_fwd = h_fwd.detach().clone()
            self._last_fwd_dist = fwd_dist

        # ---- 3b. Cache update: plan ONLY at swing onset for each foot ----
        # Left foot swing onset  → plan left-foot target (stance = right foot, sign = +1)
        # Right foot swing onset → plan right-foot target (stance = left foot, sign = -1)
        for foot_idx in range(2):
            mask = swing_onset[:, foot_idx]
            if mask.any():
                if foot_idx == 0:  # left foot
                    p_new = self._planner.plan_in_world(
                        heightmap[mask],
                        v_cmd[mask],
                        body_pos[mask, 1],
                        root_pos[mask],
                        root_quat[mask],
                        torch.ones(mask.sum(), device=env.device),
                        com_pos_w=com_pos_w[mask],
                        com_vel_w=com_vel_w[mask],
                        k=k[mask],
                    )
                    self._p_star_cache[mask, 0] = p_new
                else:  # right foot
                    p_new = self._planner.plan_in_world(
                        heightmap[mask],
                        v_cmd[mask],
                        body_pos[mask, 0],
                        root_pos[mask],
                        root_quat[mask],
                        -torch.ones(mask.sum(), device=env.device),
                        com_pos_w=com_pos_w[mask],
                        com_vel_w=com_vel_w[mask],
                        k=k[mask],
                    )
                    self._p_star_cache[mask, 1] = p_new
                self._swing_planned[mask, foot_idx] = True

                # ---- Bézier setup at swing onset (shared state) ----
                # Record lift-off position (P0)
                self._lift_off_pos[mask, foot_idx] = foot_center[mask, foot_idx]
                # Reset swing timer
                self._swing_elapsed[mask, foot_idx] = 0.0

                # Compute apex position (P1, Eq 7 + Eq 9)
                p_l = self._lift_off_pos[mask, foot_idx]  # (M, 3) lift-off
                p_f = self._p_star_cache[mask, foot_idx]  # (M, 3) target
                dz = p_f[:, 2] - p_l[:, 2]  # (M,)
                dz_abs = dz.abs()
                # Bias (Eq 7)
                bias = (0.5 + self._kappa * dz / self._planner.h_max).clamp(self._b_min, self._b_max)
                # Apex xy interpolated (Eq 7)
                apex_xy = (1.0 - bias).unsqueeze(-1) * p_l[:, :2] + bias.unsqueeze(-1) * p_f[:, :2]
                # Clearance (Eq 9)
                c = (self._c_min + self._c_scale * dz_abs).clamp(max=self._c_max)
                # Apex z (Eq 9): z_apex = 2*(max(z_l,z_f) + c) - 0.5*(z_l + z_f)
                max_z = torch.max(p_l[:, 2], p_f[:, 2])  # (M,)
                apex_z = 2.0 * (max_z + c) - 0.5 * (p_l[:, 2] + p_f[:, 2])
                self._apex_cache[mask, foot_idx] = torch.stack([apex_xy[:, 0], apex_xy[:, 1], apex_z], dim=-1)

                # u_peak (Eq 10) — where tangent is vertical
                z_l = p_l[:, 2]
                z_f = p_f[:, 2]
                z_a = apex_z
                denom = z_l - 2.0 * z_a + z_f
                u_peak = torch.where(denom.abs() > 1e-8, (z_l - z_a) / denom, 0.5 * torch.ones_like(z_l))
                self._u_peak_cache[mask, foot_idx] = u_peak

        # -- Cost-channel visualisation (full planning each frame, cache NOT updated) --
        if self._debug_vis and self._cost_visualizer is not None:
            (
                p_left_swing_vis,
                self._channels_left,
            ) = self._planner.plan_with_channels_in_world(
                heightmap,
                v_cmd,
                body_pos[:, 1],
                root_pos,
                root_quat,
                torch.ones(env.num_envs, device=env.device),
                com_pos_w=com_pos_w,
                com_vel_w=com_vel_w,
                k=k,
            )
            (
                p_right_swing_vis,
                self._channels_right,
            ) = self._planner.plan_with_channels_in_world(
                heightmap,
                v_cmd,
                body_pos[:, 0],
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
            self._last_L_nom = self._channels_left["L_nom"]  # (N, 1, 1)
            self._last_W_nom_left = self._channels_left["W_nom"]  # (N, 1, 1)  swing_leg_sign=+1 → +lp
            self._last_W_nom_right = self._channels_right["W_nom"]  # (N, 1, 1)  swing_leg_sign=-1 → -lp

        # ---- 4. Reward components (proximity + bezier, weighted internally) ----
        # ---- 4a. Swing-phase level (touchdown edge already computed in 2a) ---
        is_swinging = ~in_contact  # (N, 2) 电平: 当前子步是否在空中

        # ---- 4b. Swing elapsed timer (always updated; drives bezier tracking window) --
        self._swing_elapsed += env.step_dt * is_swinging.float()  # (N, 2)

        reward = torch.zeros(env.num_envs, device=env.device)
        # touchdown 门控 (与 td_mask 同源): 仅当该脚确有摆动计划时才视为真实触地事件。
        # 排除 reset 后传感器时间量重新累积产生的假触地 (reset 时 swing_planned 已清零)。
        # 注意: 必须在循环前计算, 循环内 4c-iii 会清除 swing_planned。
        touchdown_active = touchdown & self._swing_planned  # (N, 2)
        for foot_idx in range(2):
            # Shared swing-phase masks (both components vote on the same swing)
            td_mask = touchdown_active[:, foot_idx]  # (N,)

            # ---- 4c-i. Dense reward: Bézier per-frame tracking --------------------
            # --- Tracking reward for currently-swinging feet ----------
            tracking_mask = is_swinging[:, foot_idx] & self._swing_planned[:, foot_idx]
            if tracking_mask.any():
                # Quintic-smoothstep time warping (zero endpoint vel/acc).
                u_time_clamped = (self._swing_elapsed[tracking_mask, foot_idx] / self._T_swing).clamp(0.0, 1.0)
                u = u_time_clamped**3 * (10.0 - 15.0 * u_time_clamped + 6.0 * u_time_clamped**2)
                u_unsq = u.unsqueeze(-1)

                P0 = self._lift_off_pos[tracking_mask, foot_idx]
                P1 = self._apex_cache[tracking_mask, foot_idx]
                P2 = self._p_star_cache[tracking_mask, foot_idx]

                one_minus_u = 1.0 - u_unsq
                p_bezier = one_minus_u**2 * P0 + 2.0 * one_minus_u * u_unsq * P1 + u_unsq**2 * P2

                pos_err = foot_center[tracking_mask, foot_idx] - p_bezier
                pos_err_sq = (pos_err**2).sum(dim=-1)

                # Orientation error (optional)
                if self._sigma_d > 0.0:
                    p_dot = 2.0 * one_minus_u * (P1 - P0) + 2.0 * u_unsq * (P2 - P1)
                    p_dot_norm = torch.norm(p_dot, dim=-1, keepdim=True).clamp(min=1e-8)
                    p_dot_unit = p_dot / p_dot_norm

                    u_peak_val = self._u_peak_cache[tracking_mask, foot_idx].unsqueeze(-1)
                    pre_mask = (u >= u_peak_val.squeeze(-1) - self._delta_l_minus) & (
                        u < u_peak_val.squeeze(-1) - self._delta_l_plus
                    )
                    t_hat_pre = torch.stack([p_dot_unit[:, 2], p_dot_unit[:, 1], -p_dot_unit[:, 0]], dim=-1)
                    t_hat_pre = t_hat_pre / (torch.norm(t_hat_pre, dim=-1, keepdim=True).clamp(min=1e-8))
                    t_hat = torch.where(pre_mask.unsqueeze(-1), t_hat_pre, torch.zeros_like(p_dot_unit))
                    post_mask = (u > u_peak_val.squeeze(-1) + self._delta_r_minus) & (
                        u <= u_peak_val.squeeze(-1) + self._delta_r_plus
                    )
                    t_hat = torch.where(post_mask.unsqueeze(-1), p_dot_unit, t_hat)

                    foot_quat = body_quat[tracking_mask, foot_idx]
                    e_x = torch.tensor([1.0, 0.0, 0.0], device=foot_quat.device, dtype=foot_quat.dtype)
                    e_x = e_x.unsqueeze(0).expand(tracking_mask.sum(), -1)
                    d_hat_f = quat_apply(foot_quat, e_x)
                    ori_active = (pre_mask | post_mask).unsqueeze(-1).float()
                    ori_err_sq = ((d_hat_f - t_hat) ** 2).sum(dim=-1) * ori_active.squeeze(-1)
                else:
                    ori_err_sq = 0.0

                # Progress-gated tracking reward: u × exp(−σ_p·d² − σ_d·e_ori)
                # Multiplying by u eliminates the reward-hacking loophole:
                # at the slow-start phase (u ≈ 0) the robot gets ~0 reward
                # regardless of tracking quality → must actually advance the
                # swing to earn meaningful reward.
                tracking_quality = torch.exp(-sigma_p * pos_err_sq - self._sigma_d * ori_err_sq)
                reward[tracking_mask] += self._bezier_weight * u * tracking_quality

            # ---- 4c-ii. Sparse reward: one-shot proximity at touchdown ----------
            if self._proximity_weight > 0.0 and td_mask.any():
                dist = foot_center[td_mask, foot_idx] - self._p_star_cache[td_mask, foot_idx]  # (M, 3)
                dist_sq = (dist**2).sum(dim=-1)  # (M,)
                foot_reward = torch.exp(-sigma_p * dist_sq)  # (M,)
                reward[td_mask] += self._proximity_weight * foot_reward

            # ---- 4c-iii. Clear swing flags on touchdown (phase machine cleanup) ---
            if td_mask.any():
                self._swing_planned[td_mask, foot_idx] = False

        # ---- 4c. Velocity-condition gate -----------------------------------
        vel_cmd_full = env.command_manager.get_command("base_velocity")  # (N, 3) body frame
        has_lin_vel = torch.norm(vel_cmd_full[:, :2], dim=1) > 0.05  # (N,)
        reward = torch.where(has_lin_vel, reward, reward * 0.0)

        # ---- 4d'. Terrain-specific gating ---------------------------------
        if self._terrain_names is not None:
            self._update_terrain_mask(env)
            if self._terrain_mask is not None:
                reward = reward * self._terrain_mask

        # (边沿检测为无状态窗口触发, 无需持久化上一帧接触状态)

        # ---- 4e. Cache frame data for _debug_vis_callback -----------------
        self._last_body_pos = foot_center.clone()
        self._last_contact = in_contact.clone()
        self._last_touchdown = touchdown_active.clone()
        self._last_swing_onset = swing_onset.clone()

        # ---- Event flash timer update ------------------------------------
        if self._debug_vis and hasattr(self, "_event_timer"):
            # Reset timers on new events (flash duration in frames)
            self._event_timer[swing_onset[:, 0], 0] = 10  # left  foot swing onset
            self._event_timer[touchdown_active[:, 0], 1] = 15  # left  foot touchdown
            self._event_timer[swing_onset[:, 1], 2] = 10  # right foot swing onset
            self._event_timer[touchdown_active[:, 1], 3] = 15  # right foot touchdown
            # Decrement active timers (clamped to zero)
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
                foot_pos = self._last_body_pos[vis_mask]
                last_contact = self._last_contact[vis_mask]
            else:
                foot_pos = self._last_body_pos
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
                body_pos = self._last_body_pos[vis_mask]  # (M, 2, 3)
            else:
                p_star = self._p_star_cache
                body_pos = self._last_body_pos
            if body_pos.shape[0] > 0:
                dirs = p_star - body_pos  # (M, 2, 3)
                lengths = torch.norm(dirs, dim=-1)  # (M, 2)
                valid_mask = lengths > 0.01
                valid_flat = valid_mask.reshape(-1)  # (M*2,)
                if valid_flat.any():
                    foot_flat = body_pos.reshape(-1, 3)[valid_flat]
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
                body_pos = self._last_body_pos[vis_mask]
            else:
                event_timer = self._event_timer
                body_pos = self._last_body_pos
            if body_pos.shape[0] > 0:
                event_positions = []
                event_indices = []
                for foot_idx in range(2):
                    # Swing onset timer columns: 0 for left, 2 for right
                    swing_timer = event_timer[:, foot_idx * 2]
                    swing_active = swing_timer > 0
                    if swing_active.any():
                        event_positions.append(body_pos[swing_active, foot_idx])
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
                        event_positions.append(body_pos[td_active, foot_idx])
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

                # Convert to world frame (same rotation as plan_with_channels_in_world)
                q_conj = self._last_root_quat.clone()
                q_conj[:, 1:] *= -1.0
                nominal_world = quat_apply_inverse(q_conj, nominal_local) + self._last_root_pos  # (N, 3)

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

                q_conj = root_quat_v.clone()
                q_conj[:, 1:] *= -1.0
                fwd_xy_w = quat_apply_inverse(q_conj, fwd_local) + root_pos_v

                # Bottom at forward terrain height, top at forward terrain + step_height.
                # This ensures the line sits on the actual terrain at its XY position
                # instead of piercing through the ground when going up stairs.
                bottom_w = fwd_xy_w.clone()
                bottom_w[:, 2] = h_f
                top_w = fwd_xy_w.clone()
                top_w[:, 2] = h_f + (h_f - h_c)  # = 2*h_f - h_c

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
        self._p_star_cache[env_ids] = 0.0
        self._p_star_initialized[env_ids] = False
        # Phase-state reset: default to left-swing; randomise for symmetry.
        if isinstance(env_ids, slice):
            n = self._phase_left_swing.shape[0]
        else:
            n = env_ids.numel()
        self._phase_left_swing[env_ids] = torch.rand(n, device=self._phase_left_swing.device) > 0.5
        self._swing_planned[env_ids] = False
        self._swing_elapsed[env_ids] = 0.0
        self._lift_off_pos[env_ids] = 0.0
        self._apex_cache[env_ids] = 0.0
        self._u_peak_cache[env_ids] = 0.0
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


class ActionSmoothnessTerm(ManagerTermBase):
    """Second-order action smoothness penalty from the paper.

    Computes ``||a_t - 2*a_{t-1} + a_{t-2}||^2`` summed over the full action
    vector (num_envs, total_action_dim) and returns a per-env scalar cost.

    The cost is positive (larger = less smooth); the RewTerm weight should be
    negative (paper: ``-2.5e-3``) so it becomes a penalty.

    This is implemented as a **stateful** :class:`ManagerTermBase` (same pattern
    as :class:`FootholdReward`) so it keeps its own rolling 3-step action
    history internally and never needs to override ``env.step()``. It reads the
    raw policy output from ``env.action_manager.action`` (the pre-action-term
    buffer, updated by ``process_action`` before reward computation).

    ``total_action_dim`` is read dynamically, so it works for any action space
    (e.g. 29 joints + 1 gait-frequency = 30 for the parkour G1, or 31+1 = 32).
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._num_envs = env.num_envs
        self._device = env.device
        self._action_dim = env.action_manager.total_action_dim
        # rolling history index: 0 = current, 1 = t-1, 2 = t-2
        self._history = torch.zeros(self._num_envs, 3, self._action_dim, device=self._device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        warmup_steps: int = 0,
    ) -> torch.Tensor:
        """Compute the smoothness cost for the current step.

        Args:
            env: The environment.
            warmup_steps: Zero-out the penalty for the first ``warmup_steps``
                after a reset, while the action history is still warming up from
                zeros (avoids a spurious spike at episode start). 0 by default.
        """
        a = env.action_manager.action  # (num_envs, total_action_dim) raw policy output
        # roll the history: t-2 <- t-1, t-1 <- t, t <- new action
        self._history[:, 2] = self._history[:, 1]
        self._history[:, 1] = self._history[:, 0]
        self._history[:, 0] = a

        diff = self._history[:, 0] - 2.0 * self._history[:, 1] + self._history[:, 2]
        cost = torch.sum(diff * diff, dim=-1)  # (num_envs,)

        if warmup_steps > 0:
            # mask the first `warmup_steps` control steps after each reset
            mask = (env.episode_length_buf > warmup_steps).float()
            cost = cost * mask
        return cost

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the action history for the given envs (called on env reset)."""
        if env_ids is None:
            env_ids = slice(None)
        self._history[env_ids] = 0.0


def action_limit(env: ManagerBasedRLEnv, limit: float = 1.0) -> torch.Tensor:
    """Count the number of action dimensions outside ``[-limit, limit]``.

    The penalty reward in the paper is ``r = -w * n_lim`` where ``n_lim`` is the
    number of raw policy outputs exceeding the allowed range. This function
    returns the (positive) count; attach a negative weight (paper: ``-0.25``).

    Operates on the **raw policy output** (``env.action_manager.action``, before
    any action-term scaling / clipping), so it does not see gait-frequency that
    was already clipped to ``[0.7, 1.3]`` in the action term -- it penalises the
    pre-clipping value. If the policy head is ``tanh`` the output is naturally
    bounded and this reward is ~0; it is intended as a safety backstop for
    early-stage / unstable policies.
    """
    a = env.action_manager.action  # (num_envs, total_action_dim)
    exceeded = torch.logical_or(a > limit, a < -limit)
    return exceeded.sum(dim=-1).float()  # (num_envs,)
