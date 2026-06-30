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
    """Create a config with *num_bins* cube primitives forming a blue→red gradient,
    plus one extra bright-red cube for the selected (argmin) cell.

    Each prototype is a small cube with a distinct colour.  The gradient
    goes through cyan and yellow so all bins are visually separable.
    The final prototype (``selected``) is a slightly larger bright-red cube.
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
    # Bright-red marker for the selected (argmin) cell
    markers["selected"] = CuboidCfg(
        size=(0.06, 0.06, 0.025),
        visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    return VisualizationMarkersCfg(prim_path=base_prim_path, markers=markers)


class DCMCostVisualizer:
    """Render one DCM planner cost channel as a coloured-cube heatmap in the viewport.

    Each grid cell of the planner (pelvis-local frame) is transformed to
    world coordinates and drawn as a small coloured cube.  Colour encodes
    the cost value via a 20-bin blue→red gradient (low cost = blue).

    All parallel environments are rendered simultaneously by concatenating
    their markers into a single :meth:`VisualizationMarkers.visualize` call.
    Environments where both feet are in contact (no valid swing) are skipped.

    .. code-block:: python

        vis = DCMCostVisualizer(planner, num_envs=4096, device=env.device)
        vis.update(channels, heightmap, root_pos_w, root_quat_w, in_contact=contact)
    """

    CHANNEL_NAMES = ("Q", "E", "M", "b", "d_pos", "d_dcm", "J")

    def __init__(
        self,
        planner,
        num_envs: int,
        device: str,
        active_channel: str = "J",
    ):
        """
        Args:
            planner: A :class:`DCMFootholdPlanner` instance (used to read grid geometry).
            num_envs: Number of parallel environments (unused, for interface consistency).
            device: Torch device string.
            active_channel: Which cost channel to display (default: ``"J"``).
        """
        self._planner = planner
        self._device = device
        self._active_channel = active_channel

        # Grid coordinates from planner (pelvis-local frame, flattened)
        self._gx = planner.grid_x.reshape(-1)  # (H*W,)
        self._gy = planner.grid_y.reshape(-1)  # (H*W,)
        self._num_cells = self._gx.shape[0]

        # Create visualisation markers (20 coloured cube prototypes + 1 selected)
        # Single prim path — all environments' markers are batched in one
        # PointInstancer.  The env-to-world pose is handled per-marker in update().
        vis_cfg = _build_colored_cube_markers(
            base_prim_path="/Visuals/DCMCostMap",
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
        in_contact: torch.Tensor | None = None,
    ):
        """Render the active cost channel for *all* environments as a heatmap.

        Each environment's cost map is transformed to world coordinates and
        concatenated into a single :meth:`visualize` call, so the single
        :class:`VisualizationMarkers` draws cubes for every env simultaneously.

        Args:
            channels: Output dict from ``planner.plan_with_channels*()``.
            heightmap: Pelvis-local heightmap ``(N, H, W)`` (used only for z-values).
            root_pos_w: Pelvis world position ``(N, 3)``.
            root_quat_w: Pelvis world quaternion ``(N, 4)`` (w, x, y, z).
            in_contact: Contact state ``(N, 2)`` bool — if provided the visualiser
                hides markers for environments where *both* feet are in contact
                (no valid swing foot to show).
        """
        n_envs = channels["J"].shape[0]
        ch_name = self._active_channel

        cost = channels[ch_name]           # (N, H, W)
        if "valid" in channels:
            valid = channels["valid"]      # (N, H, W) bool
        else:
            valid = torch.ones_like(cost, dtype=torch.bool)
        if "h_safe" in channels:
            h_safe = channels["h_safe"]   # (N, H, W)
        else:
            h_safe = heightmap             # (N, H, W)

        all_world_pos = []
        all_bin_idx = []
        sel_world_pos: list[torch.Tensor] = []
        sel_bin_idx: list[torch.Tensor] = []

        for i in range(n_envs):
            # Skip envs where both feet are in contact (no swing to show)
            if in_contact is not None and i < in_contact.shape[0]:
                if in_contact[i, 0].item() and in_contact[i, 1].item():
                    continue

            c = cost[i].reshape(-1)          # (H*W,)
            v = valid[i].reshape(-1)         # (H*W,) bool
            z = h_safe[i].reshape(-1)        # (H*W,)

            mask = v
            n_valid = mask.sum().item()
            if n_valid == 0:
                continue

            # Min-max normalise → quantise to 0 … 19
            c_sel = c[mask]
            c_min, c_max = c_sel.min(), c_sel.max()
            if c_max > c_min:
                c_norm = (c_sel - c_min) / (c_max - c_min)
            else:
                c_norm = torch.zeros_like(c_sel)
            bin_idx = (c_norm * 19).long().clamp(0, 19)  # keep on original device

            # Build local positions
            # Lift markers by a small fixed offset so they float above the terrain
            # and are clearly visible (avoids being partially buried / occluded).
            local_pos = torch.stack(
                [self._gx[mask], self._gy[mask], z[mask] + 0.5], dim=-1
            )  # (n_valid, 3)

            # Rotate pelvis-local → world: p_w = R(q) @ p_local + t
            q_conj = root_quat_w[i].clone()
            q_conj[1:] *= -1.0
            world_pos = quat_apply_inverse(q_conj, local_pos) + root_pos_w[i]

            all_world_pos.append(world_pos)
            all_bin_idx.append(bin_idx)

            # ---- Selected cell (red cube) ----
            best_idx = channels.get("best_idx")  # (N,) flat index or None
            if best_idx is not None:
                bi = best_idx[i].item()
                if bi >= 0:  # valid (not low-speed override)
                    # Convert flat index to local (x, y, z)
                    sx = self._gx[bi]
                    sy = self._gy[bi]
                    sz = h_safe[i].reshape(-1)[bi]
                    sel_local = torch.tensor(
                        [sx, sy, sz + 0.5], device=self._device
                    ).unsqueeze(0)  # (1, 3)
                    # Transform to world
                    q_conj = root_quat_w[i].clone()
                    q_conj[1:] *= -1.0
                    sel_world = quat_apply_inverse(
                        q_conj, sel_local
                    ) + root_pos_w[i]
                    sel_world_pos.append(sel_world)
                    # index = 20 (the "selected" prototype)
                    sel_bin_idx.append(
                        torch.tensor([20], device=self._device, dtype=torch.long)
                    )

        if not all_world_pos:
            self._visualizer.visualize(translations=None)
            return

        # Merge heatmap markers and selected markers into one call
        all_pos = [torch.cat(all_world_pos, dim=0)]
        all_idx = [torch.cat(all_bin_idx, dim=0)]
        if sel_world_pos:
            all_pos.append(torch.cat(sel_world_pos, dim=0))
            all_idx.append(torch.cat(sel_bin_idx, dim=0))

        self._visualizer.visualize(
            translations=torch.cat(all_pos, dim=0),
            marker_indices=torch.cat(all_idx, dim=0),
        )

    def set_active_channel(self, name: str):
        """Switch the displayed cost channel.

        Valid names: ``"Q"``, ``"E"``, ``"M"``, ``"b"``,
        ``"d_pos"``, ``"d_dcm"``, ``"J"``.
        """
        if name in self.CHANNEL_NAMES:
            self._active_channel = name

    def close(self):
        """Hide all markers."""
        self._visualizer.set_visibility(False)
