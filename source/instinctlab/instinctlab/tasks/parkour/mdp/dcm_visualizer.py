"""
DCM cost-channel visualizer for Isaac Lab.

Renders DCM planner intermediate cost channels (Q, E, M, b, d_pos, d_dcm, J)
as a colored-cube heatmap in the simulation viewport using Isaac Lab's
:class:`VisualizationMarkers` (UsdGeom.PointInstancer).

Color encoding: 20-bin blue→cyan→green→yellow→red gradient.
Low cost → blue, high cost → red (per-frame min-max normalised).
"""
from __future__ import annotations

import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import CuboidCfg, PreviewSurfaceCfg
from isaaclab.utils.math import quat_apply_inverse


def _build_colored_cube_markers(
    num_bins: int = 20,
    base_prim_path: str = "/Visuals/DCMCostMap",
) -> VisualizationMarkersCfg:
    """Create a config with *num_bins* cube primitives forming a blue→red gradient.

    Each prototype is a small cube with a distinct colour.  The gradient
    goes through cyan and yellow so all bins are visually separable.
    """
    markers = {}
    for i in range(num_bins):
        t = i / max(num_bins - 1, 1)
        # blue → cyan → green → yellow → red
        if t < 0.25:
            r, g, b = 0.0, t * 4.0, 1.0
        elif t < 0.50:
            r, g, b = 0.0, 1.0, 1.0 - (t - 0.25) * 4.0
        elif t < 0.75:
            r, g, b = (t - 0.50) * 4.0, 1.0, 0.0
        else:
            r, g, b = 1.0, 1.0 - (t - 0.75) * 4.0, 0.0
        markers[f"bin_{i:02d}"] = CuboidCfg(
            size=(0.045, 0.045, 0.02),
            visual_material=PreviewSurfaceCfg(diffuse_color=(r, g, b)),
        )
    return VisualizationMarkersCfg(prim_path=base_prim_path, markers=markers)


class DCMCostVisualizer:
    """Render one DCM planner cost channel as a coloured-cube heatmap in the viewport.

    Each grid cell of the planner (pelvis-local frame) is transformed to
    world coordinates and drawn as a small coloured cube.  Colour encodes
    the cost value via a 20-bin blue→red gradient (low cost = blue).

    .. code-block:: python

        vis = DCMCostVisualizer(planner, num_envs=1, device=env.device)
        vis.update(channels, heightmap, root_pos_w, root_quat_w)
    """

    CHANNEL_NAMES = ("Q", "E", "M", "b", "d_pos", "d_dcm", "J")

    def __init__(
        self,
        planner,
        num_envs: int,
        device: str,
        active_channel: str = "J",
        env_idx: int = 0,
    ):
        """
        Args:
            planner: A :class:`DCMFootholdPlanner` instance (used to read grid geometry).
            num_envs: Number of parallel environments (unused, for interface consistency).
            device: Torch device string.
            active_channel: Which cost channel to display (default: ``"J"``).
            env_idx: Which environment index to visualise (default: ``0``).
        """
        self._planner = planner
        self._device = device
        self._active_channel = active_channel
        self._env_idx = env_idx

        # Grid coordinates from planner (pelvis-local frame, flattened)
        self._gx = planner.grid_x.reshape(-1)  # (H*W,)
        self._gy = planner.grid_y.reshape(-1)  # (H*W,)
        self._num_cells = self._gx.shape[0]

        # Create visualisation markers (20 coloured cube prototypes)
        vis_cfg = _build_colored_cube_markers(
            base_prim_path=f"/Visuals/DCMCostMap/env_{env_idx}",
        )
        self._visualizer = VisualizationMarkers(vis_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        channels: dict[str, torch.Tensor],
        heightmap: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ):
        """Render the active cost channel for env_idx as a heatmap.

        Args:
            channels: Output dict from ``planner.plan_with_channels*()``.
            heightmap: Pelvis-local heightmap ``(N, H, W)`` (used only for z-values).
            root_pos_w: Pelvis world position ``(N, 3)``.
            root_quat_w: Pelvis world quaternion ``(N, 4)`` (w, x, y, z).
        """
        idx = self._env_idx
        if idx >= channels["J"].shape[0]:
            return  # env index out of range

        ch_name = self._active_channel
        cost = channels[ch_name][idx]  # (H, W)
        # valid mask: use planner's valid if available, else all-true
        if "valid" in channels:
            valid = channels["valid"][idx]  # (H, W) bool
        else:
            valid = torch.ones_like(cost, dtype=torch.bool)
        # h_safe: use planner's safe heights if available, else raw heightmap
        if "h_safe" in channels:
            h_safe = channels["h_safe"][idx]  # (H, W)
        else:
            h_safe = heightmap[idx]  # (H, W)

        # Flatten
        cost_flat = cost.reshape(-1)
        valid_flat = valid.reshape(-1)
        z_flat = h_safe.reshape(-1)

        mask = valid_flat
        n_valid = mask.sum().item()
        if n_valid == 0:
            self._visualizer.visualize(translations=None)  # clear
            return

        # Min-max normalise → quantise to 0 … 19
        c_sel = cost_flat[mask]
        c_min, c_max = c_sel.min(), c_sel.max()
        if c_max > c_min:
            c_norm = (c_sel - c_min) / (c_max - c_min)
        else:
            c_norm = torch.zeros_like(c_sel)
        bin_idx = (c_norm * 19).long().clamp(0, 19).cpu()

        # Build world positions for valid cells
        local_pos = torch.stack(
            [self._gx[mask], self._gy[mask], z_flat[mask]], dim=-1
        )  # (n_valid, 3)

        # Rotate pelvis-local → world: p_w = R(q) @ p_local + t
        q_conj = root_quat_w[idx].clone()
        q_conj[1:] *= -1.0  # conjugate: apply same rotation but opposite direction
        world_pos = quat_apply_inverse(q_conj, local_pos) + root_pos_w[idx]

        # Render
        self._visualizer.visualize(
            translations=world_pos,
            marker_indices=bin_idx,
        )

    def set_active_channel(self, name: str):
        """Switch the displayed cost channel.

        Valid names: ``"Q"``, ``"E"``, ``"M"``, ``"b"``,
        ``"d_pos"``, ``"d_dcm"``, ``"J"``.
        """
        if name in self.CHANNEL_NAMES:
            self._active_channel = name

    def set_env_index(self, idx: int):
        """Change which environment is visualised."""
        self._env_idx = idx

    def close(self):
        """Hide all markers."""
        self._visualizer.set_visibility(False)
