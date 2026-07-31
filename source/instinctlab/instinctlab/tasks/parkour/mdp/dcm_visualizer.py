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


def clear_markers(visualizer: VisualizationMarkers, device):
    """Hide all markers of *visualizer* without raising.

    ``visualize(translations=None)`` with no marker indices raises
    ``ValueError`` while the instancer still holds instances (the empty
    call is guarded by ``num_markers == 0``).  Passing an empty
    ``marker_indices`` tensor brings the instance count to zero, which
    both hides everything and makes later calls safe again.
    """
    visualizer.visualize(
        translations=None,
        marker_indices=torch.zeros(0, dtype=torch.int, device=device),
    )


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

        cost = channels[ch_name]                       # (N, H, W)
        if "valid" in channels:
            valid = channels["valid"]                 # (N, H, W) bool
        else:
            valid = torch.ones_like(cost, dtype=torch.bool)
        if "h_safe" in channels:
            h_safe = channels["h_safe"]              # (N, H, W)
        else:
            h_safe = heightmap                        # (N, H, W)

        # Keep envs with at least one swing foot (skip both-in-contact),
        # and envs that have no valid cell at all (nothing to render).
        keep = torch.ones(n_envs, dtype=torch.bool, device=self._device)
        if in_contact is not None and in_contact.shape[0] >= n_envs:
            keep = keep & ~(in_contact[:, 0] & in_contact[:, 1])
        keep = keep & valid.reshape(n_envs, -1).any(dim=1)
        if not bool(keep.any()):
            clear_markers(self._visualizer, self._device)
            return

        cost = cost[keep]
        valid = valid[keep]
        h_safe = h_safe[keep]
        root_pos_w = root_pos_w[keep]
        root_quat_w = root_quat_w[keep]

        M, H, W = cost.shape
        HW = H * W
        c_flat = cost.reshape(M, HW)                  # (M, HW)
        v_flat = valid.reshape(M, HW)                 # (M, HW) bool
        z_flat = h_safe.reshape(M, HW)                # (M, HW)

        # Per-env min-max normalisation over valid cells (fully vectorised).
        c_min = torch.where(
            v_flat, c_flat, torch.full_like(c_flat, float("inf"))
        ).min(dim=1).values                            # (M,)
        c_max = torch.where(
            v_flat, c_flat, torch.full_like(c_flat, float("-inf"))
        ).max(dim=1).values                            # (M,)
        span = (c_max - c_min).clamp(min=1e-8)
        c_norm = ((c_flat - c_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
        bin_idx = (c_norm * 19.0).long().clamp(0, 19)  # (M, HW)

        # Pelvis-local cell positions, lifted above the terrain for visibility.
        gx_all = self._gx.unsqueeze(0).expand(M, HW)
        gy_all = self._gy.unsqueeze(0).expand(M, HW)
        local_pos = torch.stack([gx_all, gy_all, z_flat + 0.5], dim=-1)  # (M, HW, 3)

        # Rotate pelvis-local -> world for every cell in one batched call.
        q_conj = root_quat_w.clone()
        q_conj[:, 1:] *= -1.0
        q_cells = q_conj[:, None, :].expand(M, HW, 4).reshape(-1, 4)
        world_pos = quat_apply_inverse(q_cells, local_pos.reshape(-1, 3))
        world_pos = world_pos.reshape(M, HW, 3) + root_pos_w[:, None, :]  # (M, HW, 3)

        # Gather only valid cells into a flat marker array.
        world_pos_v = world_pos[v_flat]               # (K, 3)
        bin_v = bin_idx[v_flat]                       # (K,)

        # Selected (argmin) cells — bright-red cube prototype (index 20).
        sel_world = None
        best_idx = channels.get("best_idx")           # (N,) flat index or None
        if best_idx is not None:
            best_m = best_idx[keep]                   # (M,)
            ok = best_m >= 0
            if bool(ok.any()):
                row = torch.arange(M, device=self._device)
                bi = best_m.clamp(min=0)              # (M,)  (low-speed override -> -1)
                sel_z = z_flat[row, bi]               # (M,)
                sel_local = torch.stack(
                    [self._gx[bi], self._gy[bi], sel_z + 0.5], dim=-1
                )                                     # (M, 3)
                sel_world = quat_apply_inverse(q_conj, sel_local) + root_pos_w
                sel_world = sel_world[ok]             # (L, 3)

        if world_pos_v.shape[0] == 0 and (
            sel_world is None or sel_world.shape[0] == 0
        ):
            clear_markers(self._visualizer, self._device)
            return

        parts_pos = [world_pos_v]
        parts_idx = [bin_v]
        if sel_world is not None and sel_world.shape[0] > 0:
            parts_pos.append(sel_world)
            parts_idx.append(
                torch.full(
                    (sel_world.shape[0],), 20, dtype=torch.long, device=self._device
                )
            )

        self._visualizer.visualize(
            translations=torch.cat(parts_pos, dim=0),
            marker_indices=torch.cat(parts_idx, dim=0),
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
