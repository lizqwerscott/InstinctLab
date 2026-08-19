"""
DCM Foothold Planner (TACT-ful style) — minimal Isaac Lab implementation.

Computes a multi-channel terrain cost over a local pelvis heightmap and
returns the best (x, y, z) landing target per env.

Reference: TACT-ful paper §3.2. Simplified: only the best foothold is
returned (no Bézier swing, no orientation schedule). Designed to be
used as a reward anchor for RL training.

All ops are torch + F.max_pool2d / F.conv2d, fully GPU-batched.
One forward call returns p*_f for every env.

from https://arxiv.org/abs/2601.10365v1 and https://arxiv.org/abs/2601.10365v1
"""

import math
import torch
import torch.nn.functional as F

from isaaclab.utils.math import quat_apply_inverse, quat_apply_yaw, yaw_quat


# ---------------------------------------------------------------------------
# Sobel kernels (cached at planner construction)
# ---------------------------------------------------------------------------
def _make_sobel_kernels(device, dtype=torch.float32):
    """3x3 Sobel-x and Sobel-y kernels.

    NOTE: divided by 8 to normalise the gradient approximation so that a
    height change of 1 cell over 1 cell gives |g| ~ 1.  This means the
    alpha_E weight directly controls the cost per-unit-gradient.  If you
    want to match an un-normalised Sobel (paper convention), remove the
    "/ 8.0" and increase alpha_E by ~8x.
    """
    kx = (
        torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    ky = (
        torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    return kx, ky


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class DCMFootholdPlanner:
    """
    Per-env GPU-parallel DCM foothold planner.

    Input  : (num_envs, H, W) tensor of pelvis-local heights
             (NaN = invalid / ray-miss).
    Output : (num_envs, 3) tensor of best (x, y, z) in pelvis-local frame.

    Args mirror the TACT-ful paper Table 1, exposed as keyword args so you
    can sweep them without changing the class.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        grid_h: int = 25,  # rows = y direction
        grid_w: int = 37,  # cols = x direction
        cell_size: float = 0.05,
        # LIPM
        z0: float = 0.82,
        T: float = 0.45,
        lp: float = 0.20,
        # Footprint kernel (in cells)
        fp_h: int = 2,
        fp_w: int = 4,
        # Cost weights
        alpha_pos: float = 3.0,
        alpha_dcm: float = 0.5,
        alpha_E: float = 0.6,
        alpha_Q: float = 4.0,
        alpha_M: float = 6.0,
        alpha_climb: float = 1.5,
        beta: float = 2.5,
        # Velocity-aware feasibility
        h_min: float = 0.05,
        h_max: float = 0.28,
        v_star: float = 0.5,
        v_min: float = 0.05,
        # Range mask (cells within [-max_bwd_range, max_fwd_range] along x)
        max_fwd_range: float = 0.6,
        max_bwd_range: float = 0.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cell_size = cell_size
        self.fp_h = fp_h
        self.fp_w = fp_w
        self.T = T
        self.lp = lp

        # ---- Pre-compute cell-center coords in pelvis-local frame ----
        xs = (torch.arange(grid_w, device=device, dtype=torch.float32) - (grid_w - 1) / 2) * cell_size
        ys = (torch.arange(grid_h, device=device, dtype=torch.float32) - (grid_h - 1) / 2) * cell_size
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # ---- VHIP constants ----
        self.z0 = z0
        self.omega0 = math.sqrt(9.81 / z0)
        # flat-ground limit: exp(ω₀·T), used when k=None or all k≈0
        self.exp_wT_flat = math.exp(self.omega0 * T)
        self.bx_coef_flat = T / (self.exp_wT_flat - 1.0)
        self.by_coef_flat = lp / (self.exp_wT_flat - 1.0)

        # ---- Cost weights ----
        self.alpha_pos = alpha_pos
        self.alpha_dcm = alpha_dcm
        self.alpha_E = alpha_E
        self.alpha_Q = alpha_Q
        self.alpha_M = alpha_M
        self.alpha_climb = alpha_climb
        self.beta = beta

        # ---- Feasibility ----
        self.h_min = h_min
        self.h_max = h_max
        self.v_star = v_star
        self.v_min = v_min

        # ---- Range mask ----
        self.max_fwd_range = max_fwd_range
        self.max_bwd_range = max_bwd_range

        # ---- Sobel kernels ----
        self.kx, self.ky = _make_sobel_kernels(device)

    # -----------------------------------------------------------------------
    # Shared channel computation
    # -----------------------------------------------------------------------
    def _compute_dcm_params(self, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-environment DCM parameters for sloped terrain.

        Given slope k, the CoM height varies linearly with time:
            z(t) = k·t + z₀

        The LIPM natural frequency becomes time-dependent:
            ω(t) = √(g / z(t))

        The integrated frequency σ(T) over swing time T:
            σ(T) = ∫₀ᵀ ω(τ)dτ = 2√g · (√(kT+z₀) − √z₀) / k

        The DCM evolves as ξ(t) = u₀ + (ξ₀ − u₀)·exp(σ(t)) with CoP u₀ fixed at
        the stance foot during swing, so the *offset from the CoP* amplifies by
        exp(σ(T)) over the swing phase.

        When k → 0 (flat ground), σ(T) → ω₀·T where ω₀ = √(g/z₀).

        Args:
            k: (N,) per-environment slope (vertical velocity of CoM height).
               k = 0  →  flat ground
               k > 0  →  ascending (CoM rises)
               k < 0  →  descending (CoM falls)

        Returns:
            exp_sigma:  (N,)  e^{σ(T)} — amplification factor of the CoP offset (ξ₀ − u₀)
            exp_sigma_m1: (N,)  e^{σ(T)} − 1  (safe, clamped away from 0)
        """
        g = 9.81
        sqrt_g = math.sqrt(g)
        sqrt_z0 = math.sqrt(self.z0)

        # Flat ground limit value: exp(ω₀·T)
        exp_flat = self.exp_wT_flat

        # Start with flat-ground default for all environments
        exp_sigma = torch.full_like(k, exp_flat)

        # Only compute sloped formula for non-flat environments (avoid 0/0)
        non_flat = k.abs() > 1e-6
        if torch.any(non_flat):
            k_nf = k[non_flat]
            kT_z0_nf = k_nf * self.T + self.z0
            kT_z0_nf = torch.clamp(kT_z0_nf, min=1e-8)
            sqrt_kT_z0_nf = torch.sqrt(kT_z0_nf)
            sigma_T_nf = 2.0 * sqrt_g * (sqrt_kT_z0_nf - sqrt_z0) / k_nf
            exp_sigma[non_flat] = torch.exp(sigma_T_nf)

        # Clamp so exp_sigma ≠ 1 (avoid division by zero in b_nom)
        exp_sigma = torch.clamp(exp_sigma, min=1.0 + 1e-6)
        exp_sigma_m1 = exp_sigma - 1.0

        return exp_sigma, exp_sigma_m1

    # -----------------------------------------------------------------------

    def _compute_channels(
        self,
        heightmap: torch.Tensor,  # (N, H, W) pelvis-local heights, NaN = invalid
        v_cmd: torch.Tensor,  # (N, 2) commanded velocity in pelvis-local
        stance_xyz_local: torch.Tensor,  # (N, 3) stance foot in pelvis-local
        swing_leg_sign: torch.Tensor,  # (N,) +-1
        com_local: torch.Tensor | None = None,  # (N, 2) CoM (x, y) in pelvis-local
        com_vel_local: torch.Tensor | None = None,  # (N, 2) CoM velocity in pelvis-local
        k: torch.Tensor | None = None,  # (N,) per-environment slope, None = all flat
    ) -> dict[str, torch.Tensor]:
        """Compute all intermediate cost channels.

        Returns dict with keys:
            Q, E, M, b, d_pos, d_dcm, J, valid, h_safe
        each of shape (N, H, W) except valid (bool).
        """
        N = heightmap.shape[0]
        H, W = self.grid_h, self.grid_w
        device = self.device

        # ---- Validity mask ----
        valid = ~torch.isnan(heightmap)
        # Only consider cells within [-max_bwd_range, max_fwd_range] along x
        x_mask = (self.grid_x >= -self.max_bwd_range) & (self.grid_x <= self.max_fwd_range)
        valid = valid & x_mask.unsqueeze(0)

        # Q: -inf/+inf sentinels so invalid cells never win max/min.
        h_for_max = torch.where(
            valid,
            heightmap,
            torch.tensor(float("-inf"), device=device, dtype=heightmap.dtype),
        ).unsqueeze(1)
        h_for_min = torch.where(
            valid, -heightmap, torch.tensor(float("-inf"), device=device, dtype=heightmap.dtype)
        ).unsqueeze(1)

        # For M, E, and argmin z-lookup: 0-fill (neutral for dz).
        h_safe = torch.where(valid, heightmap, torch.zeros_like(heightmap))

        pad_h, pad_w = self.fp_h // 2, self.fp_w // 2

        # =================================================================
        # Channel 1: Flatness Q (Eq. 2)
        # =================================================================
        Q_max = F.max_pool2d(h_for_max, (self.fp_h, self.fp_w), stride=1, padding=(pad_h, pad_w))
        Q_neg_min = F.max_pool2d(h_for_min, (self.fp_h, self.fp_w), stride=1, padding=(pad_h, pad_w))
        Q_min = -Q_neg_min
        Q = (Q_max - Q_min).squeeze(1)[:, :H, :W]

        # =================================================================
        # Channel 2: Steepness E (Eq. 3)
        # =================================================================
        h_in = h_safe.unsqueeze(1)
        gx = F.conv2d(h_in, self.kx, padding=1)
        gy = F.conv2d(h_in, self.ky, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-6)
        E = F.max_pool2d(grad, (self.fp_h, self.fp_w), stride=1, padding=(pad_h, pad_w)).squeeze(1)[:, :H, :W]

        # =================================================================
        # Channel 3 & 4: Feasibility M and Climb bonus b (Eq. 4-5)
        # =================================================================
        vx = v_cmd[:, 0]
        vy = v_cmd[:, 1]
        vx_abs = vx.abs()
        h_eff = self.h_min + (self.h_max - self.h_min) * torch.clamp(vx_abs / self.v_star, 0.0, 1.0)
        h_eff_map = h_eff.view(N, 1, 1)

        stance_z = stance_xyz_local[:, 2]
        dz = h_safe - stance_z.view(N, 1, 1)
        abs_dz = dz.abs()

        M = torch.clamp(abs_dz - h_eff_map, min=0.0) ** 2

        climb_pos = torch.clamp(dz, min=0.0).clamp(max=h_eff_map)
        b = climb_pos * (vx_abs > self.v_min).view(N, 1, 1).float()

        # =================================================================
        # Position residual d_pos (Eq. 1 first term)
        # =================================================================
        gx_map = self.grid_x.unsqueeze(0)
        gy_map = self.grid_y.unsqueeze(0)
        vx_map = vx.view(N, 1, 1)
        vy_map = vy.view(N, 1, 1)
        sgn_map = swing_leg_sign.view(N, 1, 1).float()

        # ---- Nominal step length and width (common to d_pos and b_nom) ----
        L_nom = vx_map * self.T  # (N,1,1)  = vx·T
        W_nom = vy_map * self.T + sgn_map * self.lp  # (N,1,1)  = vy·T + (-1)ⁱ·l

        d_pos = (gx_map - L_nom) ** 2 + self.beta * (gy_map - W_nom) ** 2

        # =================================================================
        # DCM residual d_dcm (Eq. 1 second term)
        #
        # DCM dynamics on variable-height LIPM:
        #   ξ̇ = ω(t)·(ξ − u₀)   with CoP u₀ = stance foot (fixed during swing)
        #   ω(τ) = √(g / z(τ)),  z(τ) = k·τ + z₀
        #
        # General solution (homogeneous offset from the CoP amplifies):
        #   ξ(t) = u₀ + (ξ₀ − u₀)·exp(∫₀ᵗ ω(τ)dτ) = u₀ + (ξ₀ − u₀)·e^{σ(t)}
        #
        # σ(T) = 2√g · (√(kT+z₀) − √z₀) / k
        #
        # Nominal DCM offset (= desired distance from DCM target to foothold):
        #   b_nom_x = vx·T / (e^{σ(T)} − 1)
        #   b_nom_y = (-1)ⁱ·lₚ/(1+e^{σ(T)}) − (vy·T + (-1)ⁱ·lₚ)/(1−e^{σ(T)})
        #
        # DCM cost:
        #   d_dcm = ‖ξ_T − p − b_nom‖²
        # =================================================================
        # _compute_dcm_params handles both flat and sloped internally
        if k is not None:
            exp_sigma, exp_sigma_m1 = self._compute_dcm_params(k)  # both (N,)

            exp_sigma_exp = exp_sigma.view(N, 1, 1)
            exp_sigma_m1_exp = exp_sigma_m1.view(N, 1, 1)
            sgn_exp = swing_leg_sign.float().view(N, 1, 1)

            # b_nom_x = vx·T / (e^{σ} − 1) = L_nom / (e^{σ}−1)
            bx_map = L_nom / exp_sigma_m1_exp

            # b_nom_y = (-1)ⁱ·lp/(1+e^{σ}) − W_nom/(1−e^{σ})
            # Note: (1 − e^{σ}) = −(e^{σ} − 1), so:
            #   = (-1)ⁱ·lp/(1+e^{σ}) + W_nom/(e^{σ}−1)
            one_plus_exp = 1.0 + exp_sigma_exp
            by_map = (sgn_exp * self.lp) / one_plus_exp + W_nom / exp_sigma_m1_exp

            # ξ_T = u₀ + (ξ₀ − u₀)·e^{σ(T)},  u₀ = stance foot (CoP) in pelvis-local
            if com_local is not None and com_vel_local is not None:
                xi_0_x = com_local[:, 0] + com_vel_local[:, 0] / self.omega0
                xi_0_y = com_local[:, 1] + com_vel_local[:, 1] / self.omega0
                u0_x = stance_xyz_local[:, 0]
                u0_y = stance_xyz_local[:, 1]
                xi_T_x = (u0_x + (xi_0_x - u0_x) * exp_sigma).view(N, 1, 1)
                xi_T_y = (u0_y + (xi_0_y - u0_y) * exp_sigma).view(N, 1, 1)
            else:
                # Fallback: no CoM state → d_dcm reduces to position norm
                xi_T_x = gx_map + bx_map
                xi_T_y = gy_map + by_map
        else:
            # ---- k is None → all environments are flat ----
            sgn_exp = swing_leg_sign.float().view(N, 1, 1)

            # b_nom_x = L_nom / (e^{ω₀·T}−1)   (= vx · bx_coef_flat)
            bx_map = L_nom / (self.exp_wT_flat - 1.0)

            one_plus_exp_flat = 1.0 + self.exp_wT_flat
            by_map = (sgn_exp * self.lp) / one_plus_exp_flat + W_nom / (self.exp_wT_flat - 1.0)

            # ξ_T = u₀ + (ξ₀ − u₀)·e^{ω₀·T},  u₀ = stance foot (CoP) in pelvis-local
            if com_local is not None and com_vel_local is not None:
                xi_0_x = com_local[:, 0] + com_vel_local[:, 0] / self.omega0
                xi_0_y = com_local[:, 1] + com_vel_local[:, 1] / self.omega0
                u0_x = stance_xyz_local[:, 0]
                u0_y = stance_xyz_local[:, 1]
                xi_T_x = (u0_x + (xi_0_x - u0_x) * self.exp_wT_flat).view(N, 1, 1)
                xi_T_y = (u0_y + (xi_0_y - u0_y) * self.exp_wT_flat).view(N, 1, 1)
            else:
                xi_T_x = gx_map + bx_map
                xi_T_y = gy_map + by_map

        d_dcm = (xi_T_x - gx_map - bx_map) ** 2 + (xi_T_y - gy_map - by_map) ** 2

        # =================================================================
        # Total cost (Eq. 1)
        # =================================================================
        J = (
            self.alpha_pos * d_pos
            + self.alpha_dcm * d_dcm
            + self.alpha_E * E
            + self.alpha_Q * Q
            + self.alpha_M * M
            - self.alpha_climb * b
        )

        J = torch.where(valid, J, torch.full_like(J, float("inf")))

        return {
            "Q": Q,
            "E": E,
            "M": M,
            "b": b,
            "d_pos": d_pos,
            "d_dcm": d_dcm,
            "J": J,
            "valid": valid,
            "h_safe": h_safe,
            "L_nom": L_nom,
            "W_nom": W_nom,
        }

    # -----------------------------------------------------------------------
    # Core: argmin over the local heightmap
    # -----------------------------------------------------------------------
    def plan(
        self,
        heightmap: torch.Tensor,  # (N, H, W) pelvis-local heights, NaN = invalid
        v_cmd: torch.Tensor,  # (N, 2) commanded velocity in pelvis-local
        stance_xyz_local: torch.Tensor,  # (N, 3) stance foot in pelvis-local
        swing_leg_sign: torch.Tensor,  # (N,) +-1
        com_local: torch.Tensor | None = None,  # (N, 2) CoM (x, y) in pelvis-local
        com_vel_local: torch.Tensor | None = None,  # (N, 2) CoM velocity in pelvis-local
        k: torch.Tensor | None = None,  # (N,) per-environment slope, None = all flat
    ) -> torch.Tensor:
        """Returns p_star (N, 3): best (x, y, z) in pelvis-local frame."""
        channels = self._compute_channels(
            heightmap,
            v_cmd,
            stance_xyz_local,
            swing_leg_sign,
            com_local,
            com_vel_local,
            k=k,
        )
        return self._argmin(channels["J"], channels["h_safe"], v_cmd[:, 0].abs(), stance_xyz_local)[0]

    def _argmin(
        self,
        J: torch.Tensor,  # (N, H, W) total cost
        h_safe: torch.Tensor,  # (N, H, W) safe heights
        vx_abs: torch.Tensor,  # (N,) absolute forward velocity
        stance_xyz_local: torch.Tensor,  # (N, 3) stance foot
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Argmin over J, return (p_star, best_idx) in pelvis-local.

        p_star: (N, 3) best foothold position.
        best_idx: (N,) flat index into H*W grid of the selected cell.
        """
        N, H, W = J.shape
        J_flat = J.view(N, -1)
        best_idx = torch.argmin(J_flat, dim=-1)
        i = best_idx // W
        j = best_idx % W

        p_star_x = self.grid_x[i, j]
        p_star_y = self.grid_y[i, j]
        p_star_z = h_safe[torch.arange(N, device=self.device), i, j]

        # Low-velocity and no-valid-cell fallback.
        low_speed = vx_abs < self.v_min
        has_valid_cell = torch.isfinite(J_flat).any(dim=-1)
        use_stance_fallback = low_speed | ~has_valid_cell
        p_star_x = torch.where(use_stance_fallback, stance_xyz_local[:, 0], p_star_x)
        p_star_y = torch.where(use_stance_fallback, stance_xyz_local[:, 1], p_star_y)
        p_star_z = torch.where(use_stance_fallback, stance_xyz_local[:, 2], p_star_z)

        p_star = torch.stack([p_star_x, p_star_y, p_star_z], dim=-1)

        # Fallbacks have no selected heightmap cell.
        best_idx = torch.where(use_stance_fallback, -1, best_idx)

        return p_star, best_idx

    def plan_with_channels(
        self,
        heightmap: torch.Tensor,  # (N, H, W) pelvis-local heights, NaN = invalid
        v_cmd: torch.Tensor,  # (N, 2) commanded velocity in pelvis-local
        stance_xyz_local: torch.Tensor,  # (N, 3) stance foot in pelvis-local
        swing_leg_sign: torch.Tensor,  # (N,) +-1
        com_local: torch.Tensor | None = None,  # (N, 2) CoM (x, y) in pelvis-local
        com_vel_local: torch.Tensor | None = None,  # (N, 2) CoM velocity in pelvis-local
        k: torch.Tensor | None = None,  # (N,) per-environment slope, None = all flat
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (p_star, channels) where channels contains all intermediate costs.

        p_star: (N, 3) best foothold in pelvis-local frame.
        channels: dict with keys Q, E, M, b, d_pos, d_dcm, J, valid, h_safe.
        """
        channels = self._compute_channels(
            heightmap,
            v_cmd,
            stance_xyz_local,
            swing_leg_sign,
            com_local,
            com_vel_local,
            k=k,
        )
        p_star, best_idx = self._argmin(channels["J"], channels["h_safe"], v_cmd[:, 0].abs(), stance_xyz_local)
        channels["best_idx"] = best_idx
        return p_star, channels

    def plan_with_channels_in_world(
        self,
        heightmap: torch.Tensor,  # (N, H, W) pelvis-local heights
        v_cmd_yaw_local: torch.Tensor,  # (N, 2) yaw-local commanded velocity
        stance_xyz_world: torch.Tensor,  # (N, 3) stance foot world pos
        root_pos_w: torch.Tensor,  # (N, 3) pelvis world pos
        root_quat_w: torch.Tensor,  # (N, 4) pelvis world quat (w,x,y,z)
        swing_leg_sign: torch.Tensor,  # (N,)
        com_pos_w: torch.Tensor | None = None,  # (N, 3) CoM world pos
        com_vel_w: torch.Tensor | None = None,  # (N, 3) CoM world vel
        k: torch.Tensor | None = None,  # (N,) per-environment slope
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute a world-frame foothold from yaw-local velocity and return cost channels.

        Returns (p_world, channels) where channels has keys:
            Q, E, M, b, d_pos, d_dcm, J, valid, h_safe
        all in pelvis-local frame (shape N, H, W).
        """
        root_yaw_quat_w = yaw_quat(root_quat_w)

        # -- Stance foot: world -> yaw-local --
        stance_local = quat_apply_inverse(root_yaw_quat_w, stance_xyz_world - root_pos_w)

        # -- CoM: world -> pelvis-local (optional) --
        com_local = com_vel_local = None
        if com_pos_w is not None and com_vel_w is not None:
            com_local = quat_apply_inverse(root_yaw_quat_w, com_pos_w - root_pos_w)[:, :2]
            com_vel_local = quat_apply_inverse(root_yaw_quat_w, com_vel_w)[:, :2]

        # -- Plan in pelvis-local frame --
        p_local, channels = self.plan_with_channels(
            heightmap,
            v_cmd_yaw_local,
            stance_local,
            swing_leg_sign,
            com_local,
            com_vel_local,
            k=k,
        )

        # -- Rotate back: yaw-local -> world --
        p_world = quat_apply_yaw(root_quat_w, p_local) + root_pos_w
        return p_world, channels

    # -----------------------------------------------------------------------
    # Convenience: world-frame positions, yaw-local velocity
    # -----------------------------------------------------------------------
    def plan_in_world(
        self,
        heightmap: torch.Tensor,  # (N, H, W) pelvis-local heights
        v_cmd_yaw_local: torch.Tensor,  # (N, 2) yaw-local commanded velocity
        stance_xyz_world: torch.Tensor,  # (N, 3) stance foot world pos
        root_pos_w: torch.Tensor,  # (N, 3) pelvis world pos
        root_quat_w: torch.Tensor,  # (N, 4) pelvis world quat (w,x,y,z)
        swing_leg_sign: torch.Tensor,  # (N,)
        com_pos_w: torch.Tensor | None = None,  # (N, 3) CoM world pos
        com_vel_w: torch.Tensor | None = None,  # (N, 3) CoM world vel
        k: torch.Tensor | None = None,  # (N,) per-environment slope
    ) -> torch.Tensor:
        """Compute a world-frame foothold from yaw-local velocity."""
        root_yaw_quat_w = yaw_quat(root_quat_w)

        # -- Stance foot: world -> yaw-local --
        stance_local = quat_apply_inverse(root_yaw_quat_w, stance_xyz_world - root_pos_w)

        # -- CoM: world -> pelvis-local (optional) --
        com_local = com_vel_local = None
        if com_pos_w is not None and com_vel_w is not None:
            com_local = quat_apply_inverse(root_yaw_quat_w, com_pos_w - root_pos_w)[:, :2]
            com_vel_local = quat_apply_inverse(root_yaw_quat_w, com_vel_w)[:, :2]

        # -- Plan in pelvis-local frame --
        p_local = self.plan(
            heightmap,
            v_cmd_yaw_local,
            stance_local,
            swing_leg_sign,
            com_local,
            com_vel_local,
            k=k,
        )

        # -- Rotate back: pelvis-local -> world (yaw-only, matching heightmap) --
        p_world = quat_apply_yaw(root_quat_w, p_local) + root_pos_w
        return p_world
