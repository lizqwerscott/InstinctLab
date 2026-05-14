# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn.functional as F

class DepthNoise(torch.nn.Module):
    def __init__(self,
        focal_length,
        baseline,
        min_depth,
        max_depth,
        filter_size=3,
        inlier_thred_range=(0.01, 0.05),
        prob_range=(0.4, 0.6),
        invalid_disp=1e7,
        # ---- NYST augmentation params (per Eqs. 3-5 in the paper) ----
        enable_random_conv: bool = False,
        random_conv_range: tuple = (-0.05, 0.05),
        enable_depth_dependent_noise: bool = True,
        depth_noise_coef_range: tuple = (-0.03, 0.03),
        enable_perlin: bool = False,
        perlin_base_res: tuple = (3, 4),
        perlin_octaves: int = 5,
        perlin_persistence: float = 0.5,
        perlin_coef_range: tuple = (-0.02, 0.02),
        enable_scale_randomization: bool = True,
        scale_range: tuple = (0.90, 1.10),
        enable_pixel_failures: bool = True,
        p_zero_pixel: float = 0.001,
        p_max_pixel: float = 0.001,
    ):
        """
        A Simply PyTorch module to add realistic noise to depth images.

        Args:
            focal_length (float): Focal length of the camera (in pixels).
            baseline (float): Baseline distance between stereo cameras (in meters).
            min_depth (float): Minimum depth value after clamping.
            max_depth (float): Maximum depth value after clamping.
            filter_size (int): Kernel size for local mean disparity computation. (tuning based on image resolution)
            inlier_thred_range (tuple): Threshold range for normalized disparity differences.
            prob_range (tuple): Probability range for matching pixels.
            invalid_disp (float): Invalid disparity

        """
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.invalid_disp = invalid_disp
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = filter_size

        self.enable_random_conv = enable_random_conv
        self.random_conv_range = random_conv_range
        self.enable_depth_dependent_noise = enable_depth_dependent_noise
        self.depth_noise_coef_range = depth_noise_coef_range
        self.enable_perlin = enable_perlin
        self.perlin_base_res = perlin_base_res
        self.perlin_octaves = perlin_octaves
        self.perlin_persistence = perlin_persistence
        self.perlin_coef_range = perlin_coef_range
        self.enable_scale_randomization = enable_scale_randomization
        self.scale_range = scale_range
        self.enable_pixel_failures = enable_pixel_failures
        self.p_zero_pixel = p_zero_pixel
        self.p_max_pixel = p_max_pixel

        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer('weights', weights.view(1, 1, filter_size, filter_size))
        self.register_buffer('substitutes', substitutes.view(1, 1, filter_size, filter_size))
        
        
    def _compute_weights(self, filter_size):
        """
        Compute weights and substitutes for disparity filtering.

        Args:
            filter_size (int): Kernel size for local mean disparity computation.
        """
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing='ij')
        sqr_radius = x_filter ** 2 + y_filter ** 2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()
        
        return weights, substitutes

    def filter_disparity(self, disparity):
        """
        Filter the disparity map using local mean disparity.
        
        Args:
            disparity (torch.Tensor): Input disparity map tensor of shape (B, C, H, W).
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        prob = torch.rand(B, 1, 1, 1, device=device) * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]
        random_mask = (torch.rand(B, 1, H, W, device=device) < prob)

        # Compute mean disparity
        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)

        # Compute differences
        differences = torch.abs(disparity - weighted_disparity)

        # Normalize differences based on current image statistics for consistent thresholding
        differences_flat = differences.view(B, -1)  # Flatten spatial dimensions
        mean_diff = torch.mean(differences_flat, dim=1, keepdim=True)
        std_diff = torch.std(differences_flat, dim=1, keepdim=True) + 1e-6  # Add epsilon to avoid division by zero

        # Normalize differences: (diff - mean) / std, then shift to [0, 1] range approximately
        normalized_differences_flat = (differences_flat - mean_diff) / std_diff
        normalized_differences = normalized_differences_flat.view_as(differences)

        # Use parameter-based threshold on normalized differences
        threshold = torch.rand(B, 1, 1, 1, device=device) * (self.inlier_thred_range[1] - self.inlier_thred_range[0]) + self.inlier_thred_range[0]
        update_mask = (normalized_differences < threshold) & random_mask

        # Compute output value: round with 1/32 precision
        disparity = torch.round(disparity * 32.0) / 32.0

        # Update output disparity
        output_disparity = torch.where(update_mask, disparity, output_disparity)

        # Apply substitutes to fill neighboring pixels
        filled_values = F.conv2d(update_mask.float() * disparity, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)

        return output_disparity

    def _apply_random_conv(self, depth: torch.Tensor) -> torch.Tensor:
        """
        NYST Step 2: 局部 3×3 随机卷积,模拟光学畸变.
        每个环境独立采样卷积核.
        """
        B = depth.shape[0]
        device = depth.device
        low, high = self.random_conv_range
        
        # 每个 batch 独立采样 W,加上 I (恒等核)
        W = torch.empty(B, 1, 3, 3, device=device).uniform_(low, high)
        I = torch.zeros(B, 1, 3, 3, device=device)
        I[:, 0, 1, 1] = 1.0
        kernel = W + I  # (B, 1, 3, 3)
        
        # 用 grouped conv 实现 "每个 batch 不同核"
        # 把 batch 维度 reshape 成 groups
        # depth: (B, 1, H, W) → (1, B, H, W)
        depth_reshaped = depth.transpose(0, 1)  # (1, B, H, W)
        kernel_reshaped = kernel.view(B, 1, 3, 3)  # (B, 1, 3, 3) for B groups
        
        out = F.conv2d(depth_reshaped, kernel_reshaped, padding=1, groups=B)
        return out.transpose(0, 1)  # 回到 (B, 1, H, W)
    
    def _apply_depth_dependent_noise(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Paper Eq. (3): depth-dependent Gaussian noise.
            sigma(d) = |c0 + c1*d + c2*d^2|, c_i ~ U(-0.03, 0.03) per environment.
        """
        B = depth.shape[0]
        device = depth.device
        low, high = self.depth_noise_coef_range

        c0 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        c1 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        c2 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)

        sigma = torch.abs(c0 + c1 * depth + c2 * depth ** 2)
        noise = torch.randn_like(depth) * sigma
        return depth + noise

    def _fade(self, t: torch.Tensor) -> torch.Tensor:
        # Perlin's quintic fade: 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _perlin_octave(self, B: int, H: int, W: int, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        """
        Generate a single Perlin-noise octave at resolution (H, W) using a (gh+1, gw+1)
        grid of random unit gradient vectors per batch. Returns shape (B, 1, H, W).
        """
        # Random unit gradient vectors at each grid corner, per batch
        angles = torch.rand(B, gh + 1, gw + 1, device=device) * (2.0 * torch.pi)
        grad_x = torch.cos(angles)
        grad_y = torch.sin(angles)

        # Pixel coordinates expressed in grid units (each grid cell covers H/gh rows, W/gw cols)
        ys = torch.linspace(0.0, float(gh), H, device=device)
        xs = torch.linspace(0.0, float(gw), W, device=device)
        # Clamp to avoid the very last sample falling exactly on the boundary (index gh / gw)
        ys = torch.clamp(ys, max=float(gh) - 1e-6)
        xs = torch.clamp(xs, max=float(gw) - 1e-6)

        y0 = torch.floor(ys).long()  # (H,)
        x0 = torch.floor(xs).long()  # (W,)
        y1 = y0 + 1
        x1 = x0 + 1
        dy = (ys - y0.float()).view(1, H, 1)  # (1, H, 1)
        dx = (xs - x0.float()).view(1, 1, W)  # (1, 1, W)

        # Gather gradients at the four corners of each pixel's containing cell — shape (B, H, W)
        gy00_y = grad_y[:, y0][:, :, x0]
        gy00_x = grad_x[:, y0][:, :, x0]
        gy10_y = grad_y[:, y0][:, :, x1]
        gy10_x = grad_x[:, y0][:, :, x1]
        gy01_y = grad_y[:, y1][:, :, x0]
        gy01_x = grad_x[:, y1][:, :, x0]
        gy11_y = grad_y[:, y1][:, :, x1]
        gy11_x = grad_x[:, y1][:, :, x1]

        # Dot products between corner gradient and offset-to-corner
        ones_h = torch.ones(1, H, 1, device=device)
        ones_w = torch.ones(1, 1, W, device=device)
        d00 = gy00_x * dx + gy00_y * dy
        d10 = gy10_x * (dx - ones_w) + gy10_y * dy
        d01 = gy01_x * dx + gy01_y * (dy - ones_h)
        d11 = gy11_x * (dx - ones_w) + gy11_y * (dy - ones_h)

        # Smooth interpolation
        u = self._fade(dx)  # (1, 1, W)
        v = self._fade(dy)  # (1, H, 1)
        ix0 = d00 + u * (d10 - d00)
        ix1 = d01 + u * (d11 - d01)
        noise = ix0 + v * (ix1 - ix0)  # (B, H, W) in roughly [-sqrt(2)/2, sqrt(2)/2]
        return noise.unsqueeze(1)  # (B, 1, H, W)

    def _apply_perlin_noise(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Paper Eq. (4): multi-octave Perlin structured noise modulated by depth.
            n_perlin = sum_{o=0..octaves-1} persistence^o * P(2^o * u, 2^o * v)
            sigma_p(d) = |cp0 + cp1*d + cp2*d^2|,  cp_i ~ U(perlin_coef_range)
            d_tilde = d + sigma_p(d) * n_perlin
        """
        B: int = depth.shape[0]
        H: int = depth.shape[2]
        W: int = depth.shape[3]
        device = depth.device
        gh0: int = int(self.perlin_base_res[0])
        gw0: int = int(self.perlin_base_res[1])

        noise = torch.zeros(B, 1, H, W, device=device)
        amplitude = 1.0
        amp_sum = 0.0
        for o in range(self.perlin_octaves):
            scale: int = 1 << o
            gh_cand: int = gh0 * scale
            gw_cand: int = gw0 * scale
            # Cap the grid so it can't be finer than the image
            gh: int = gh_cand if gh_cand < H else H
            gw: int = gw_cand if gw_cand < W else W
            noise = noise + amplitude * self._perlin_octave(B, H, W, gh, gw, device)
            amp_sum += amplitude
            amplitude *= self.perlin_persistence

        # Normalize so the summed octaves stay roughly in [-1, 1] regardless of octave count
        noise = noise / max(amp_sum, 1e-6)

        low, high = self.perlin_coef_range
        cp0 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        cp1 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        cp2 = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        sigma_p = torch.abs(cp0 + cp1 * depth + cp2 * depth ** 2)
        return depth + sigma_p * noise

    def _apply_scale_randomization(self, depth: torch.Tensor) -> torch.Tensor:
        """
        NYST Step 5: 全局深度缩放,模拟标定漂移.
            d' = s * d, s ~ U(0.90, 1.10)
        """
        B = depth.shape[0]
        device = depth.device
        low, high = self.scale_range
        s = torch.empty(B, 1, 1, 1, device=device).uniform_(low, high)
        return depth * s
    
    def _apply_pixel_failures(self, depth: torch.Tensor) -> torch.Tensor:
        """
        NYST Step 6: 随机像素失效(死像素 → 0,饱和像素 → max_depth).
        """
        device = depth.device
        rand_vals = torch.rand_like(depth)
        # 死像素
        zero_mask = rand_vals < self.p_zero_pixel
        # 饱和像素(注意范围别和死像素重叠)
        max_mask = (rand_vals >= self.p_zero_pixel) & (
            rand_vals < (self.p_zero_pixel + self.p_max_pixel)
        )
        depth = depth.masked_fill(zero_mask, 0.0)
        depth = depth.masked_fill(max_mask, self.max_depth)
        return depth
    

    def forward(self, depth) -> torch.Tensor:
        # correct input shape
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1) # add channel dimension
        
        # check dimension (B, 1, H, W)
        assert depth.shape[1] == 1, "Input depth tensor must have shape (B, 1, H, W)."
        assert len(depth.shape) == 4, "Input depth tensor must have shape (B, 1, H, W)."
        
        # Clamp the depth values
        depth = torch.clamp(depth, min=1. / self.invalid_disp)
    
        # Step 1: Convert depth to disparity
        disparity = self.focal_length * self.baseline / depth

        # Step 2: Filter the disparity map
        filtered_disparity = self.filter_disparity(disparity)

        # Step 3: Recompute depth from disparity
        depth = self.focal_length * self.baseline / filtered_disparity

        hole_mask = (depth < self.min_depth) | (depth > self.max_depth)
        # Zero out invalid pixels before convolutional / multiplicative ops so they don't bleed
        depth = depth.masked_fill(hole_mask, 0.0)

        # ===== NYST augmentation pipeline =====
        # Eq. (3): depth-dependent Gaussian noise
        if self.enable_depth_dependent_noise:
            depth = self._apply_depth_dependent_noise(depth)

        # Eq. (4): structured Perlin noise modulated by depth
        if self.enable_perlin:
            depth = self._apply_perlin_noise(depth)

        # Eq. (5): random-conv optical distortion
        if self.enable_random_conv:
            depth = self._apply_random_conv(depth)

        # Calibration drift: global depth scaling
        if self.enable_scale_randomization:
            depth = self._apply_scale_randomization(depth)

        # Pixel failures (dead / saturated)
        if self.enable_pixel_failures:
            depth = self._apply_pixel_failures(depth)

        # Final clean-up: holes stay zero; out-of-range values are not measurable
        depth = depth.masked_fill(hole_mask, 0.0)
        depth = torch.where(depth > self.max_depth, torch.zeros_like(depth), depth)
        depth = torch.where((depth < self.min_depth) & (depth != 0.0), torch.zeros_like(depth), depth)

        return depth
    
    
# Example usage
if __name__ == "__main__":
    import cv2
    import yaml
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from dataloader import DepthImageDataset
    from torch.utils.data import DataLoader
    
    from network.image_utils import RandomCropDownsample
    
    def normalize_depth(depth, min_depth, max_depth, is_log):
        depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
        depth = torch.clamp(depth, min_depth, max_depth) # Clamp the depth values
        depth = torch.log(depth + 1.0) if is_log else depth
        return depth
    
    
    config_path = '/home/fanyang/Projets/world_model_pretrain/config/pretrain.yaml'
    
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    config = load_config(config_path)
    dataloader_config = config.get('dataloader', {})
    training_config = config.get('training', {})
    
    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    focal_length = dataloader_config.get('focal_length', 50.0)
    baseline = dataloader_config.get('baseline', 0.12)
    
    # Detect whether or not is at cluster
    is_cluster = False
    if 'cluster' in os.getcwd():
        print("Running on cluster.")
        is_cluster = True
    else:
        print("Running on local machine.")

    depth_dataset = DepthImageDataset(dataloader_config, is_cluster)
    data_loader = DataLoader(depth_dataset, batch_size=dataloader_config.get('batch_size', 256), shuffle=dataloader_config.get('shuffle', True), num_workers=dataloader_config.get('num_workers', 4))

    # Get a single batch from the dataloader
    depth_batch = next(iter(data_loader))
    depth = depth_batch
    batch_size = depth.shape[0]
    
    normalize_depth_fn = lambda x: normalize_depth(x, min_depth, max_depth, is_log=training_config['log_space'])
    
    depth_noise = DepthNoise(focal_length=focal_length,
                             baseline=baseline, 
                             min_depth=min_depth,
                             max_depth=max_depth)
    depth_noise = torch.jit.script(depth_noise)
    
    # Normalize the depth values
    depth = normalize_depth_fn(depth)

    # Add noise to the depth images
    noisy_depth = depth_noise(depth)
    
    # visualization resolution x4 of the original depth image
    H_orig, W_orig = depth.shape[1], depth.shape[2]
    H_vis, W_vis = H_orig * 5, W_orig * 5
    
    for i in range(batch_size):
        clip_depth = depth[i].clip(0.1, 10.0)
        disp_depth = clip_depth.squeeze().detach().cpu().numpy()
        disp_nosie_depth = noisy_depth[i].squeeze().detach().cpu().numpy()
        # enlarge the resolution for better visualization
        disp_depth = cv2.resize(disp_depth, (W_vis, H_vis))
        disp_nosie_depth = cv2.resize(disp_nosie_depth, (W_vis, H_vis))
        # normalize the depth values to 0-1
        disp_depth = (disp_depth - disp_depth.min()) / (disp_depth.max() - disp_depth.min() + 1e-7)
        disp_nosie_depth = (disp_nosie_depth - disp_nosie_depth.min()) / (disp_nosie_depth.max() - disp_nosie_depth.min())
        
        # Merge the images horizontally
        merged_image = cv2.hconcat([disp_depth, disp_nosie_depth])
        
        # Display the merged image
        cv2.imshow("Depth and Noisy Depth Image", merged_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break
        cv2.destroyAllWindows()
    
    print("Noisy depth shape:", noisy_depth.shape)


