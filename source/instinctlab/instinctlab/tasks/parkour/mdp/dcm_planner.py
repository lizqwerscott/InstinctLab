"""
DCM Foothold Planner (TACT-ful style) — minimal Isaac Lab implementation.

Computes a multi-channel terrain cost over a local pelvis heightmap and
returns the best (x, y, z) landing target per env.

Reference: TACT-ful paper §3.2. Simplified: only the best foothold is
returned (no Bézier swing, no orientation schedule). Designed to be
used as a reward anchor for RL training.

All ops are torch + F.max_pool2d / F.conv2d, fully GPU-batched.
One forward call returns p*_f for every env.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel kernels (cached at planner construction)
# ---------------------------------------------------------------------------
def _make_sobel_kernels(device, dtype=torch.float32):
    """3x3 Sobel-x and Sobel-y kernels for normalized gradients."""
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
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
        grid_h: int = 25,        # rows = y direction
        grid_w: int = 37,        # cols = x direction
        cell_size: float = 0.05,
        # LIPM
        z0: float = 0.90,
        T: float = 0.45,
        lp: float = 0.20,
        # Footprint kernel (in cells)
        fp_h: int = 2,
        fp_w: int = 5,
        # Cost weights
        alpha_pos:   float = 1.0,
        alpha_dcm:   float = 0.5,
        alpha_E:     float = 0.6,
        alpha_Q:     float = 4.0,
        alpha_M:     float = 6.0,
        alpha_climb: float = 1.5,
        beta:        float = 2.5,
        # Velocity-aware feasibility
        h_min: float = 0.05,
        h_max: float = 0.28,
        v_star: float = 0.5,
        v_min: float = 0.05,
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
        # Indexing convention: (i, j) where i = row = y, j = col = x.
        xs = (torch.arange(grid_w, device=device, dtype=torch.float32)
              - (grid_w - 1) / 2) * cell_size            # (W,)
        ys = (torch.arange(grid_h, device=device, dtype=torch.float32)
              - (grid_h - 1) / 2) * cell_size            # (H,)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        # ---- LIPM constants (Eq. 5 / b_nom) ----
        self.omega0 = math.sqrt(9.81 / z0)
        self.exp_wT = math.exp(self.omega0 * T)
        self.bx_coef = T / (self.exp_wT - 1.0)           # multiplies vx
        self.by_coef = lp / (1.0 + self.exp_wT)          # multiplies sign(f)

        # ---- Cost weights ----
        self.alpha_pos   = alpha_pos
        self.alpha_dcm   = alpha_dcm
        self.alpha_E     = alpha_E
        self.alpha_Q     = alpha_Q
        self.alpha_M     = alpha_M
        self.alpha_climb = alpha_climb
        self.beta        = beta

        # ---- Feasibility ----
        self.h_min  = h_min
        self.h_max  = h_max
        self.v_star = v_star
        self.v_min  = v_min

        # ---- Sobel kernels (used for steepness) ----
        self.kx, self.ky = _make_sobel_kernels(device)

    # -----------------------------------------------------------------------
    # Core: argmin over the local heightmap
    # -----------------------------------------------------------------------
    def plan(
        self,
        heightmap: torch.Tensor,         # (N, H, W) pelvis-local heights, NaN = invalid
        v_cmd: torch.Tensor,             # (N, 2) commanded velocity (vx, vy)
        stance_xyz_local: torch.Tensor,  # (N, 3) stance foot in pelvis-local
        swing_leg_sign: torch.Tensor,    # (N,) ±1
    ) -> torch.Tensor:
        """
        Returns p_star (N, 3): best (x, y, z) in pelvis-local frame.
        """
        N = heightmap.shape[0]
        H, W = self.grid_h, self.grid_w
        device = self.device

        # ---- Mask invalid cells: replace NaN with a very negative value
        # so that max_pool ignores them naturally.
        valid = ~torch.isnan(heightmap)
        h_safe = torch.where(valid, heightmap,
                             torch.tensor(-1.0e3, device=device))
        h_in = h_safe.unsqueeze(1)                       # (N, 1, H, W)

        # =================================================================
        # Channel 1: Flatness Q (Eq. 2)
        #   Q_i = max(z) - min(z) over the footprint-sized kernel
        # =================================================================
        pad_h, pad_w = self.fp_h // 2, self.fp_w // 2
        Q_max = F.max_pool2d(h_in, (self.fp_h, self.fp_w), stride=1,
                             padding=(pad_h, pad_w))
        # min via -max(-h)
        Q_min = -F.max_pool2d(-h_in, (self.fp_h, self.fp_w), stride=1,
                              padding=(pad_h, pad_w))
        Q = (Q_max - Q_min).squeeze(1)[:, :H, :W]          # (N, H, W)  clamp to original grid size

        # =================================================================
        # Channel 2: Steepness E (Eq. 3)
        #   E_i = max_{j in N_i} sqrt(g_x^2 + g_y^2)
        # =================================================================
        gx = F.conv2d(h_in, self.kx, padding=1)
        gy = F.conv2d(h_in, self.ky, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        E = F.max_pool2d(grad, (self.fp_h, self.fp_w), stride=1,
                         padding=(pad_h, pad_w)).squeeze(1)[:, :H, :W]

        # =================================================================
        # Channel 3 & 4: Feasibility M and Climb bonus b (Eq. 4-5)
        #   Use velocity-aware effective max step height
        # =================================================================
        vx = v_cmd[:, 0]                                 # (N,)
        vy = v_cmd[:, 1]
        vx_abs = vx.abs()
        h_eff = self.h_min + (self.h_max - self.h_min) * torch.clamp(
            vx_abs / self.v_star, 0.0, 1.0)              # (N,)
        h_eff_map = h_eff.view(N, 1, 1)

        stance_z = stance_xyz_local[:, 2]                # (N,)
        dz = h_safe - stance_z.view(N, 1, 1)            # (N, H, W)
        abs_dz = dz.abs()

        # M: quadratic overshoot penalty above h_eff
        M = torch.clamp(abs_dz - h_eff_map, min=0.0) ** 2

        # b: capped upward-step bonus, gated on minimum speed
        climb_pos = torch.clamp(dz, min=0.0).clamp(max=h_eff_map)
        b = climb_pos * (vx_abs > self.v_min).view(N, 1, 1).float()

        # =================================================================
        # Position residual d_pos and DCM residual d_dcm (Eq. 1 terms)
        # =================================================================
        gx_map = self.grid_x.unsqueeze(0)                # (1, H, W)
        gy_map = self.grid_y.unsqueeze(0)
        vx_map = vx.view(N, 1, 1)
        vy_map = vy.view(N, 1, 1)
        sgn_map = swing_leg_sign.view(N, 1, 1).float()

        # d_pos: asymmetric — penalize lateral more than sagittal (β=2.5)
        d_pos = (gx_map - vx_map * self.T) ** 2 + self.beta * (
            gy_map - (vy_map * self.T + sgn_map * self.lp)) ** 2

        # Steady-state nominal step offset b_nom
        bx = vx * self.bx_coef                           # (N,)
        by = swing_leg_sign.float() * self.by_coef       # (N,)
        bx_map = bx.view(N, 1, 1)
        by_map = by.view(N, 1, 1)
        d_dcm = (gx_map - bx_map) ** 2 + (gy_map - by_map) ** 2

        # =================================================================
        # Total cost (Eq. 1)
        # =================================================================
        J = (self.alpha_pos   * d_pos +
             self.alpha_dcm   * d_dcm +
             self.alpha_E     * E +
             self.alpha_Q     * Q +
             self.alpha_M     * M -
             self.alpha_climb * b)

        # Mask invalid cells
        J = torch.where(valid, J, torch.full_like(J, float('inf')))

        # =================================================================
        # Argmin
        # =================================================================
        J_flat = J.view(N, -1)
        best_idx = torch.argmin(J_flat, dim=-1)          # (N,)
        i = best_idx // W                                 # row -> y
        j = best_idx %  W                                 # col -> x

        # Index into pre-computed grid (and the safe heightmap for z)
        p_star_x = self.grid_x[i, j]                      # (N,)
        p_star_y = self.grid_y[i, j]                      # (N,)
        p_star_z = h_safe[torch.arange(N, device=device), i, j]

        # =================================================================
        # Low-velocity override: at near-standstill, freeze the target to
        # the current stance foot (Eq. 5 fallback in the paper)
        # =================================================================
        low_speed = vx_abs < self.v_min
        p_star_x = torch.where(low_speed, stance_xyz_local[:, 0], p_star_x)
        p_star_y = torch.where(low_speed, stance_xyz_local[:, 1], p_star_y)
        p_star_z = torch.where(low_speed, stance_xyz_local[:, 2], p_star_z)

        return torch.stack([p_star_x, p_star_y, p_star_z], dim=-1)

    # -----------------------------------------------------------------------
    # Convenience: world-frame input / output
    # -----------------------------------------------------------------------
    def plan_in_world(
        self,
        heightmap: torch.Tensor,            # (N, H, W) pelvis-local heights
        v_cmd: torch.Tensor,                # (N, 2) world-frame velocity
        stance_xyz_world: torch.Tensor,     # (N, 3) stance foot world pos
        root_pos_w: torch.Tensor,           # (N, 3) pelvis world pos
        swing_leg_sign: torch.Tensor,       # (N,)
    ) -> torch.Tensor:
        """Compute best foothold in world frame."""
        stance_local = stance_xyz_world - root_pos_w
        p_local = self.plan(heightmap, v_cmd, stance_local, swing_leg_sign)
        return p_local + root_pos_w

