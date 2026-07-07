from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import omni.kit.app
import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from instinctlab.tasks.parkour.mdp.dcm_planner import DCMFootholdPlanner
from instinctlab.tasks.parkour.mdp.dcm_visualizer import DCMCostVisualizer

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
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
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
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
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


class FootholdProximityReward(ManagerTermBase):
    """Touchdown reward for tracking DCM foothold targets.

    At swing onset a target foothold is computed using the DCM planner.
    At touchdown the L2 distance between the actual foot position and
    the previously planned target is computed, producing a one-shot
    Gaussian proximity reward per swing phase.

    Config params (resolved by the reward manager before __init__):
        asset_cfg (SceneEntityCfg): robot body config filtered to foot links.
        sensor_cfg (SceneEntityCfg): contact_forces sensor filtered to foot links.
        heightmap_sensor_cfg (SceneEntityCfg): heightmap sensor in scene.

    Call-time params (passed via RewTerm.params):
        sigma_p (float): Gaussian sharpness for proximity reward.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset_cfg = cfg.params["asset_cfg"]
        self._sensor_cfg = cfg.params["sensor_cfg"]
        self._heightmap_sensor_cfg = cfg.params["heightmap_sensor_cfg"]

        # Planner
        self._planner = DCMFootholdPlanner(
            num_envs=env.num_envs, device=env.device,
        )
        # Per-foot caching: [left, right] order (as returned by body_ids)
        # Lazily initialised with actual foot positions on first __call__ to
        # avoid rendering spheres at the world origin (0,0,0).
        self._p_star_cache = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self._p_star_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Previous-frame filtered contact state (used for rising-edge detection).
        # Initialised all-True so the first swing_onset is reliable.
        self._was_in_contact = torch.ones(env.num_envs, 2, dtype=torch.bool, device=env.device)

        # ---- Phase-state machine for swing-leg tracking -------------------
        # True  = left  leg is the swing leg (right leg is stance)
        # False = right leg is the swing leg (left  leg is stance)
        # Initialised to left-swing by default; reset() can randomise.
        self._phase_left_swing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        # Exponential-moving-average filter for raw contact signals.
        # Smoothing prevents spurious phase switches from force noise.
        self._contact_filtered = torch.ones(env.num_envs, 2, dtype=torch.float32, device=env.device)

        # Flag per foot indicating p_star_cache was set by a real plan (not lazy-init).
        # Set True at swing onset, reset False at touchdown (reward fired once).
        self._swing_planned = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

        # Resolve foot order by name (for correct left/right assignment)
        foot_names: list[str] = env.scene[self._asset_cfg.name].data.body_names
        self._foot_order: list[str] = [foot_names[i] for i in self._asset_cfg.body_ids]

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
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.0, 1.0)
                        ),
                    ),
                    "right": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0)
                        ),
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
            self._debug_vis_handle = (
                app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
            )

        # ---- Debug: print status ----
        print(f"[FootholdProximityReward] debug_vis={self._debug_vis}, "
              f"visualizer={'created' if self._foothold_visualizer is not None else 'None'},"
              f" num_envs={env.num_envs}, device={env.device}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        sigma_p: float = 10.0,
        debug_vis: bool = False,
        asset_cfg=None,
        sensor_cfg=None,
        heightmap_sensor_cfg=None,
    ) -> torch.Tensor:
        """Compute foothold proximity reward.

        Returns (N,) tensor: sum of exp(-sigma_p * dist²) per swinging foot.
        """
        asset = env.scene[self._asset_cfg.name]

        # ---- 1. Foot positions & contact (raw) --------------------------
        body_pos = asset.data.body_pos_w[:, self._asset_cfg.body_ids]  # (N, 2, 3)

        contact_sensor: ContactSensor = env.scene.sensors[self._sensor_cfg.name]
        net_force = contact_sensor.data.net_forces_w_history          # (N, hist, n_bodies_all)
        contact_norm = torch.norm(
            net_force[:, -1, self._sensor_cfg.body_ids], dim=-1       # (N, 2)
        )
        in_contact = contact_norm > 1.0                               # (N, 2) raw

        # ---- 2a. Contact-signal EMA filter (avoids oscillation) ----------
        contact_alpha = 0.85
        self._contact_filtered = (
            contact_alpha * self._contact_filtered
            + (1.0 - contact_alpha) * in_contact.float()
        )
        in_contact_smooth = self._contact_filtered > 0.3              # (N, 2)

        # ---- 2b. Swing-onset detection (rising edge on filtered signal) ---
        swing_onset = (~in_contact_smooth) & self._was_in_contact      # (N, 2)

        # ---- 2c. Phase-state update --------------------------------------
        # When a foot loses contact it becomes the new swing leg.
        #   swing_onset[:, 0] == True  → left foot just lifted  → phase_left=True
        #   swing_onset[:, 1] == True  → right foot just lifted → phase_left=False
        envs_to_swap = swing_onset[:, 0] | swing_onset[:, 1]
        new_phase = swing_onset[:, 0]  # True if left onset, False if right onset
        self._phase_left_swing = torch.where(
            envs_to_swap, new_phase, self._phase_left_swing
        )

        # ---- 3. Common quantities (shared by planning & reward) ----------
        root_pos = asset.data.root_pos_w                               # (N, 3)
        root_quat = asset.data.root_quat_w                             # (N, 4) w,x,y,z
        v_cmd = env.command_manager.get_command("base_velocity")[:, :2]  # (N, 2) world
        heightmap = self._get_heightmap(env, root_pos)                 # (N, 25, 37)

        # Approximate CoM state from root (pelvis) state.
        com_pos_w = root_pos                                          # (N, 3)
        com_vel_w = asset.data.root_lin_vel_w                          # (N, 3)

        # ---- 3a. Lazy-init p_star_cache with actual foot positions -------
        newly_uninitialized = ~self._p_star_initialized
        if newly_uninitialized.any():
            self._p_star_cache[newly_uninitialized, 0] = body_pos[newly_uninitialized, 0]
            self._p_star_cache[newly_uninitialized, 1] = body_pos[newly_uninitialized, 1]
            self._p_star_initialized[newly_uninitialized] = True

        k = 0.15 / 0.45  # stair height / max stair height to climb
        k = k * torch.ones(env.num_envs, device=env.device)

        # ---- 3b. Cache update: plan ONLY at swing onset for each foot ----
        # Left foot swing onset  → plan left-foot target (stance = right foot, sign = -1)
        # Right foot swing onset → plan right-foot target (stance = left foot, sign = +1)
        for foot_idx in range(2):
            mask = swing_onset[:, foot_idx]
            if mask.any():
                if foot_idx == 0:  # left foot
                    p_new = self._planner.plan_in_world(
                        heightmap[mask], v_cmd[mask], body_pos[mask, 1],
                        root_pos[mask], root_quat[mask],
                        -torch.ones(mask.sum(), device=env.device),
                        com_pos_w=com_pos_w[mask], com_vel_w=com_vel_w[mask],
                        k=k[mask],
                    )
                    self._p_star_cache[mask, 0] = p_new
                else:  # right foot
                    p_new = self._planner.plan_in_world(
                        heightmap[mask], v_cmd[mask], body_pos[mask, 0],
                        root_pos[mask], root_quat[mask],
                        torch.ones(mask.sum(), device=env.device),
                        com_pos_w=com_pos_w[mask], com_vel_w=com_vel_w[mask],
                        k=k[mask],
                    )
                    self._p_star_cache[mask, 1] = p_new
                self._swing_planned[mask, foot_idx] = True

        # -- Cost-channel visualisation (full planning each frame, cache NOT updated) --
        if self._debug_vis and self._cost_visualizer is not None:
            p_left_swing_vis, self._channels_left = self._planner.plan_with_channels_in_world(
                heightmap, v_cmd, body_pos[:, 1], root_pos, root_quat,
                -torch.ones(env.num_envs, device=env.device),
                com_pos_w=com_pos_w, com_vel_w=com_vel_w,
                k=k,
            )
            p_right_swing_vis, self._channels_right = self._planner.plan_with_channels_in_world(
                heightmap, v_cmd, body_pos[:, 0], root_pos, root_quat,
                torch.ones(env.num_envs, device=env.device),
                com_pos_w=com_pos_w, com_vel_w=com_vel_w,
                k=k,
            )
            self._last_heightmap = heightmap
            self._last_root_pos = root_pos
            self._last_root_quat = root_quat

        # ---- 4. Reward: one-shot at touchdown (landing) --------------------
        # ---- 4a. Touchdown detection (rising edge: not-in-contact -> in-contact) -
        touchdown = in_contact_smooth & (~self._was_in_contact)  # (N, 2)

        # ---- 4b. Touchdown proximity reward (once per swing per foot) ------
        reward = torch.zeros(env.num_envs, device=env.device)
        for foot_idx in range(2):
            mask = touchdown[:, foot_idx] & self._swing_planned[:, foot_idx]
            if mask.any():
                dist = body_pos[mask, foot_idx] - self._p_star_cache[mask, foot_idx]  # (M, 3)
                dist_sq = (dist ** 2).sum(dim=-1)                                      # (M,)
                reward[mask] += torch.exp(-sigma_p * dist_sq)                          # (M,)
                # Clear flag so this reward fires only once per swing phase
                self._swing_planned[mask, foot_idx] = False

        # ---- 4c. Velocity-condition gate -----------------------------------
        vel_cmd_full = env.command_manager.get_command("base_velocity")  # (N, 3) body frame
        has_lin_vel = torch.norm(vel_cmd_full[:, :2], dim=1) > 0.05      # (N,)
        reward = torch.where(has_lin_vel, reward, reward * 0.0)

        # ---- 4d. Persist filtered contact for next frame -----------------
        self._was_in_contact = in_contact_smooth

        return reward  # (N,)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------
    def _debug_vis_callback(self, event):
        """Render cached foothold target markers + DCM cost heatmap every frame."""
        # ---- Foothold spheres ----
        if self._foothold_visualizer is not None:
            n = self._p_star_cache.shape[0]
            if n > 0:
                # _p_star_cache: (N, 2, 3) -> (2, N, 3) -> (2*N, 3)
                poses = self._p_star_cache.permute(1, 0, 2).reshape(-1, 3)
                marker_indices = torch.zeros(2 * n, dtype=torch.int, device=self._p_star_cache.device)
                marker_indices[n:] = 1  # right foot -> green
                self._foothold_visualizer.visualize(poses, marker_indices=marker_indices)

        # ---- DCM cost heatmap ----
        if self._cost_visualizer is None:
            return
        if self._channels_left is None or self._channels_right is None:
            return

        # For each environment, pick the channel dict that corresponds to
        # the swinging foot (left-swing  → channels_left,  right-swing → channels_right).
        # The visualiser's update() already skips envs where both feet are
        # in contact, so we pass in_contact along.
        in_contact = self._was_in_contact  # (N, 2) bool

        # Build a merged channels dict: pick left or right data per-environment.
        # Both *_left and *_right come from plan_with_channels_in_world which
        # returns (N, H, W)-shaped tensors for each key.
        merged: dict[str, torch.Tensor] = {}
        for key in self._channels_left:
            left_val = self._channels_left[key]    # (N, H, W) or (N,)
            right_val = self._channels_right[key]  # (N, H, W) or (N,)
            # left foot in contact  → stance on left  → right leg is swinging
            # right foot in contact → stance on right → left leg is swinging
            # Use right-val when left foot is in contact (right is swinging)
            # Use left-val  when left foot is swinging
            if left_val.dim() == 1:
                # Scalar-per-env channels (e.g. best_idx: (N,))
                merged[key] = torch.where(
                    in_contact[:, 0],   # (N,)
                    right_val,          # (N,)
                    left_val,           # (N,)
                )
            else:
                merged[key] = torch.where(
                    in_contact[:, 0:1, None],  # (N, 1, 1) — left foot in contact?
                    right_val,   # yes → right is swinging
                    left_val,    # no  → left is swinging
                )

        self._cost_visualizer.update(
            channels=merged,
            heightmap=self._last_heightmap,
            root_pos_w=self._last_root_pos,
            root_quat_w=self._last_root_quat,
            in_contact=in_contact,
        )


    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset per-env caches (called by RewardManager on env reset)."""
        if env_ids is None:
            env_ids = slice(None)
        self._p_star_cache[env_ids] = 0.0
        self._p_star_initialized[env_ids] = False
        self._was_in_contact[env_ids] = True
        # Phase-state reset: default to left-swing; randomise for symmetry.
        if isinstance(env_ids, slice):
            n = self._phase_left_swing.shape[0]
        else:
            n = env_ids.numel()
        self._phase_left_swing[env_ids] = torch.rand(n, device=self._phase_left_swing.device) > 0.5
        self._contact_filtered[env_ids] = 1.0
        self._swing_planned[env_ids] = False

    # ------------------------------------------------------------------
    def _get_heightmap(self, env: "ManagerBasedRLEnv", root_pos: torch.Tensor) -> torch.Tensor:
        """Return (N, 25, 37) pelvis-local terrain heights (NaN = ray missed).

        The z-component of (hit_world - root_pos) gives the height
        relative to the pelvis origin.  This is frame-rotation-invariant
        (pure z differencing), so no yaw handling is needed here.
        """
        sensor = env.scene[self._heightmap_sensor_cfg.name]
        hits_w = sensor.data.ray_hits_w                                 # (N, num_rays, 3)
        num_rays = hits_w.shape[1]
        H, W = 25, num_rays // 25
        # Relative height (world z - pelvis z)
        z_rel = (hits_w[..., 2] - root_pos[:, 2].unsqueeze(1)).view(-1, H, W)
        # Mark ray-miss: hit world-z far below any reasonable terrain
        missed = hits_w.view(-1, H, W, 3)[..., 2] < -100.0
        return torch.where(missed, torch.full_like(z_rel, float("nan")), z_rel)
