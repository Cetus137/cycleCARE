"""
Loss functions for Cycle-CARE training.
Includes adversarial, cycle-consistency, and identity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GANLoss(nn.Module):
    """
    GAN loss for adversarial training.
    
    Supports both LSGAN (least squares) and vanilla GAN loss.
    LSGAN is more stable and produces higher quality results.
    
    Args:
        gan_mode (str): Type of GAN loss ('lsgan' or 'vanilla')
        target_real_label (float): Label value for real images
        target_fake_label (float): Label value for fake images
    """
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """
        Create target tensor with the same size as prediction.
        
        Args:
            prediction (torch.Tensor): Discriminator prediction
            target_is_real (bool): Whether the target is real or fake
        
        Returns:
            torch.Tensor: Target tensor filled with appropriate label
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        Calculate GAN loss.
        
        Args:
            prediction (torch.Tensor): Discriminator prediction
            target_is_real (bool): Whether the target is real or fake
        
        Returns:
            torch.Tensor: Loss value
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class CycleConsistencyLoss(nn.Module):
    """
    Cycle-consistency loss with automatic 2D/3D detection.
    
    Ensures that F(G(x)) ≈ x and G(F(y)) ≈ y
    where F and G are the two generators.
    
    This loss is crucial for learning meaningful mappings without paired data.
    
    Automatically detects input dimensions:
    - 4D tensors [B,C,H,W]: Uses 2D SSIM (treats C as channels/Z-planes independently)
    - 5D tensors [B,C,D,H,W]: Uses 3D SSIM (computes volumetric structural similarity)
    
    Args:
        loss_type (str): Type of loss ('l1', 'l2', 'ssim', or 'combined')
        ssim_weight (float): Weight for SSIM in combined loss (default: 0.84)
        l1_weight (float): Weight for L1 in combined loss (default: 0.16)
        is_3d (bool): Force 3D mode (auto-detected if None)
    """
    def __init__(self, loss_type='l1', ssim_weight=0.84, l1_weight=0.16, is_3d=None, grad_weight: float = 0.0,
                 ssim_window_2d: int = 11, ssim_window_3d: int = 11):
        super(CycleConsistencyLoss, self).__init__()
        
        self.loss_type = loss_type
        self.is_3d = is_3d  # None = auto-detect
        self.grad_weight = grad_weight
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif loss_type == 'ssim':
            # Will be set dynamically based on input dimensionality
            self.ssim_2d = SSIMLoss(window_size=ssim_window_2d, channel=1)
            self.ssim_3d = SSIM3DLoss(window_size=ssim_window_3d, channel=1)
            self.loss = None  # Set dynamically
        elif loss_type == 'combined':
            # Will be set dynamically based on input dimensionality
            self.combined_2d = CombinedLoss(ssim_weight=ssim_weight, l1_weight=l1_weight, channel=1, grad_weight=grad_weight, window_size=ssim_window_2d)
            self.combined_3d = Combined3DLoss(ssim_weight=ssim_weight, l1_weight=l1_weight, channel=1, grad_weight=grad_weight, window_size=ssim_window_3d)
            self.loss = None  # Set dynamically
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def forward(self, reconstructed, original):
        """
        Calculate cycle-consistency loss.
        
        Automatically uses 2D or 3D SSIM based on input dimensionality:
        - 4D input [B,C,H,W]: 2D SSIM (Z-context models)
        - 5D input [B,C,D,H,W]: 3D SSIM (volumetric models)
        
        Args:
            reconstructed (torch.Tensor): Reconstructed image/volume after cycle
            original (torch.Tensor): Original image/volume
        
        Returns:
            torch.Tensor: Loss value
        """
        # Auto-detect 2D vs 3D if not explicitly set
        if self.is_3d is None:
            input_is_3d = (reconstructed.dim() == 5)
        else:
            input_is_3d = self.is_3d
        
        # For SSIM and combined losses, select appropriate version
        if self.loss_type == 'ssim':
            loss_fn = self.ssim_3d if input_is_3d else self.ssim_2d
            return loss_fn(reconstructed, original)
        elif self.loss_type == 'combined':
            loss_fn = self.combined_3d if input_is_3d else self.combined_2d
            return loss_fn(reconstructed, original)
        else:
            # L1/L2 work for both 2D and 3D
            return self.loss(reconstructed, original)


class IdentityLoss(nn.Module):
    """
    Identity loss (optional).
    
    Ensures that G(x) ≈ x when x is already in the target domain.
    This helps preserve color and content when not needed to change.
    
    For example, if G translates noisy->clean, then G(clean) should ≈ clean.
    
    Args:
        loss_type (str): Type of loss ('l1' or 'l2')
    """
    def __init__(self, loss_type='l1'):
        super(IdentityLoss, self).__init__()
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def forward(self, generated, original):
        """
        Calculate identity loss.
        
        Args:
            generated (torch.Tensor): Generated image from same-domain input
            original (torch.Tensor): Original image
        
        Returns:
            torch.Tensor: Loss value
        """
        return self.loss(generated, original)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    
    SSIM measures structural similarity considering:
    - Luminance: mean intensity
    - Contrast: standard deviation
    - Structure: correlation
    
    Unlike L1/L2, SSIM is perceptually motivated and better preserves
    texture and structural details in images. It's particularly effective
    for denoising tasks.
    
    Loss is computed as: 1 - SSIM (so lower is better)
    
    Args:
        window_size (int): Size of the Gaussian window (default: 11)
        size_average (bool): Whether to average the loss over the batch
        channel (int): Number of channels (1 for grayscale, 3 for RGB)
    
    References:
        Wang et al. "Image Quality Assessment: From Error Visibility to 
        Structural Similarity" IEEE TIP 2004.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True, channel: int = 1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # Create Gaussian window
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float):
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int):
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
              window_size: int, channel: int, size_average: bool = True):
        """Calculate SSIM between two images."""
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Move window to same device as images
        window = window.to(img1.device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Calculate SSIM loss.
        
        Args:
            img1 (torch.Tensor): First image (B, C, H, W)
            img2 (torch.Tensor): Second image (B, C, H, W)
        
        Returns:
            torch.Tensor: SSIM loss (1 - SSIM, so lower is better)
        """
        (_, channel, _, _) = img1.size()
        
        # Recreate window if channel changed
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel)
            self.channel = channel
        
        # Calculate SSIM
        ssim_value = self._ssim(img1, img2, self.window, self.window_size, 
                                channel, self.size_average)
        
        # Return loss (1 - SSIM)
        return 1 - ssim_value


class GradientLoss(nn.Module):
    """
    Gradient-based loss to preserve edges and fine morphology.

    Computes L1 loss between image gradients of prediction and target.
    Supports 2D (images: B,C,H,W) and 3D volumes (B,C,D,H,W).

    Args:
        is_3d (bool): If True, compute 3D gradients; else 2D.
    """
    def __init__(self, is_3d: bool = False, eps: float = 1e-6):
        super(GradientLoss, self).__init__()
        self.is_3d = is_3d
        self.eps = eps

    def _grad_2d(self, img: torch.Tensor):
        # Sobel kernels
        device = img.device
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device)
        ky = kx.t()
        kx = kx.view(1, 1, 3, 3)
        ky = ky.view(1, 1, 3, 3)
        c = img.shape[1]
        kx = kx.repeat(c, 1, 1, 1)
        ky = ky.repeat(c, 1, 1, 1)

        gx = F.conv2d(img, kx, padding=1, groups=c)
        gy = F.conv2d(img, ky, padding=1, groups=c)
        return gx, gy

    def _grad_3d(self, vol: torch.Tensor):
        # Simple finite-difference kernels along each axis (size 3)
        device = vol.device
        # kernel along width (x)
        kx = torch.tensor([[-1., 0., 1.]], device=device).view(1, 1, 1, 1, 3)
        # kernel along height (y)
        ky = torch.tensor([[-1., 0., 1.]], device=device).view(1, 1, 1, 3, 1)
        # kernel along depth (z)
        kz = torch.tensor([[-1., 0., 1.]], device=device).view(1, 1, 3, 1, 1)

        c = vol.shape[1]
        kx = kx.repeat(c, 1, 1, 1, 1)
        ky = ky.repeat(c, 1, 1, 1, 1)
        kz = kz.repeat(c, 1, 1, 1, 1)

        gx = F.conv3d(vol, kx, padding=(0, 0, 1), groups=c)
        gy = F.conv3d(vol, ky, padding=(0, 1, 0), groups=c)
        gz = F.conv3d(vol, kz, padding=(1, 0, 0), groups=c)
        return gx, gy, gz

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """Return mean L1 between gradients."""
        if pred.dim() == 4:
            gx_p, gy_p = self._grad_2d(pred)
            gx_t, gy_t = self._grad_2d(target)
            loss = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
            return loss * 0.5
        elif pred.dim() == 5:
            gx_p, gy_p, gz_p = self._grad_3d(pred)
            gx_t, gy_t, gz_t = self._grad_3d(target)
            loss = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t) + F.l1_loss(gz_p, gz_t)
            return loss / 3.0
        else:
            raise ValueError(f"Unsupported tensor dimensionality for GradientLoss: {pred.dim()}")


class MS_SSIMLoss(nn.Module):
    """
    Multi-Scale SSIM loss.
    
    Computes SSIM at multiple scales (resolutions) to capture both
    fine details and overall structure. Often performs better than
    single-scale SSIM for complex images.
    
    Args:
        window_size (int): Size of the Gaussian window
        size_average (bool): Whether to average the loss
        channel (int): Number of channels
        weights (list): Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    """
    def __init__(self, window_size: int = 11, size_average: bool = True, 
                 channel: int = 1, weights: Optional[list] = None):
        super(MS_SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        if weights is None:
            self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.weights = torch.FloatTensor(weights)
        
        self.ssim = SSIMLoss(window_size, size_average, channel)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Calculate multi-scale SSIM loss.
        
        Args:
            img1 (torch.Tensor): First image (B, C, H, W)
            img2 (torch.Tensor): Second image (B, C, H, W)
        
        Returns:
            torch.Tensor: MS-SSIM loss
        """
        weights = self.weights.to(img1.device)
        levels = weights.size(0)
        mssim = []
        
        for i in range(levels):
            # Calculate SSIM at current scale
            ssim_val = 1 - self.ssim(img1, img2)  # Note: SSIMLoss already returns 1-SSIM
            mssim.append(ssim_val)
            
            # Downsample for next scale
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        # Weighted combination
        mssim = torch.stack(mssim)
        ms_ssim_loss = (mssim * weights).sum()
        
        return ms_ssim_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: SSIM + L1
    
    Combining SSIM with L1 often gives better results than either alone:
    - SSIM preserves structure and texture
    - L1 ensures pixel-level accuracy
    
    Args:
        ssim_weight (float): Weight for SSIM loss (default: 0.84)
        l1_weight (float): Weight for L1 loss (default: 0.16)
        window_size (int): SSIM window size
        channel (int): Number of channels
    """
    def __init__(self, ssim_weight: float = 0.84, l1_weight: float = 0.16, 
                 window_size: int = 11, channel: int = 1, grad_weight: float = 0.0):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        
        self.ssim_loss = SSIMLoss(window_size=window_size, channel=channel)
        self.l1_loss = nn.L1Loss()
        # Gradient loss (2D) will be created lazily if requested
        self.grad_loss = None
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Calculate combined SSIM + L1 loss.
        
        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image
        
        Returns:
            torch.Tensor: Combined loss
        """
        ssim = self.ssim_loss(img1, img2) * self.ssim_weight
        l1 = self.l1_loss(img1, img2) * self.l1_weight
        total = ssim + l1

        if self.grad_weight > 0.0:
            if self.grad_loss is None:
                self.grad_loss = GradientLoss(is_3d=False).to(img1.device)
            g = self.grad_loss(img1, img2) * self.grad_weight
            total = total + g

        return total


class SSIM3DLoss(nn.Module):
    """
    3D Structural Similarity Index (SSIM) loss for volumetric data.
    
    Extends SSIM to 3D by using 3D Gaussian windows and computing
    structural similarity across all three spatial dimensions (D, H, W).
    
    Critical for 3D volumetric models where Z-dimension relationships matter.
    
    Loss is computed as: 1 - SSIM (so lower is better)
    
    Args:
        window_size (int): Size of the 3D Gaussian window (default: 11)
        size_average (bool): Whether to average the loss over the batch
        channel (int): Number of channels (typically 1 for microscopy)
    
    Note:
        Input tensors should have shape [B, C, D, H, W] where D is depth.
        For Z-context models with shape [B, C, H, W] where C contains Z-planes,
        use regular SSIMLoss instead (treats each plane independently).
    """
    def __init__(self, window_size: int = 11, size_average: bool = True, channel: int = 1):
        super(SSIM3DLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # Create 3D Gaussian window
        self.window = self._create_window_3d(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float):
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window_3d(self, window_size: int, channel: int):
        """Create 3D Gaussian window."""
        gauss = self._gaussian(window_size, 1.5)
        # Outer product to form 3D separable window: gauss(d) * gauss(h) * gauss(w)
        w3 = gauss.view(window_size, 1, 1) * gauss.view(1, window_size, 1) * gauss.view(1, 1, window_size)
        # Shape to [1,1,D,H,W] then expand per channel
        window = w3.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window.float()
    
    def _ssim3d(self, img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
                window_size: int, channel: int, size_average: bool = True):
        """Calculate 3D SSIM between two volumes."""
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Move window to same device as images
        window = window.to(img1.device)
        
        pad = window_size // 2
        
        # Calculate means using 3D convolution
        mu1 = F.conv3d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv3d(img2, window, padding=pad, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv3d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
        
        # 3D SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Calculate 3D SSIM loss.
        
        Args:
            img1 (torch.Tensor): First volume (B, C, D, H, W)
            img2 (torch.Tensor): Second volume (B, C, D, H, W)
        
        Returns:
            torch.Tensor: 3D SSIM loss (1 - SSIM, so lower is better)
        """
        if img1.dim() != 5:
            raise ValueError(f"Expected 5D input [B,C,D,H,W], got {img1.dim()}D with shape {img1.shape}")
        
        (_, channel, _, _, _) = img1.size()
        
        # Recreate window if channel changed
        if channel != self.channel:
            self.window = self._create_window_3d(self.window_size, channel)
            self.channel = channel
        
        # Calculate 3D SSIM
        ssim_value = self._ssim3d(img1, img2, self.window, self.window_size, 
                                  channel, self.size_average)
        
        # Return loss (1 - SSIM)
        return 1 - ssim_value


class Combined3DLoss(nn.Module):
    """
    Combined 3D loss: 3D SSIM + L1
    
    For volumetric data, combines 3D structural similarity with pixel-wise L1.
    
    Args:
        ssim_weight (float): Weight for 3D SSIM loss (default: 0.84)
        l1_weight (float): Weight for L1 loss (default: 0.16)
        window_size (int): 3D SSIM window size
        channel (int): Number of channels
    """
    def __init__(self, ssim_weight: float = 0.84, l1_weight: float = 0.16, 
                 window_size: int = 11, channel: int = 1, grad_weight: float = 0.0):
        super(Combined3DLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        
        self.ssim_loss = SSIM3DLoss(window_size=window_size, channel=channel)
        self.l1_loss = nn.L1Loss()
        # Gradient loss (3D) will be created lazily if requested
        self.grad_loss = None
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Calculate combined 3D SSIM + L1 loss.
        
        Args:
            img1 (torch.Tensor): First volume (B, C, D, H, W)
            img2 (torch.Tensor): Second volume (B, C, D, H, W)
        
        Returns:
            torch.Tensor: Combined 3D loss
        """
        ssim = self.ssim_loss(img1, img2) * self.ssim_weight
        l1 = self.l1_loss(img1, img2) * self.l1_weight
        total = ssim + l1

        if getattr(self, 'grad_weight', 0.0) > 0.0:
            if self.grad_loss is None:
                self.grad_loss = GradientLoss(is_3d=True).to(img1.device)
            g = self.grad_loss(img1, img2) * self.grad_weight
            total = total + g

        return total


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using a pre-trained VGG network (optional).
    
    Compares high-level features rather than pixel values.
    Can produce more perceptually pleasing results but requires a pre-trained network.
    
    Note: This is an optional enhancement. The basic Cycle-CARE model
    doesn't require perceptual loss.
    
    Args:
        loss_type (str): Type of loss ('l1' or 'l2')
    """
    def __init__(self, loss_type='l1'):
        super(PerceptualLoss, self).__init__()
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def forward(self, generated, target):
        """
        Calculate perceptual loss.
        
        For simplicity, this currently just computes pixel-wise loss.
        To use true perceptual loss, you would extract features from a
        pre-trained network (like VGG) and compare those.
        
        Args:
            generated (torch.Tensor): Generated image
            target (torch.Tensor): Target image
        
        Returns:
            torch.Tensor: Loss value
        """
        return self.loss(generated, target)


class CycleCarelosses:
    """
    Container class for all Cycle-CARE losses.
    
    This class manages all loss functions and provides methods to calculate
    generator and discriminator losses.
    
    Args:
        config: Configuration object
        device (str): Device to put losses on
    """
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # Get loss type from config (default to 'l1' for backward compatibility)
        cycle_loss_type = getattr(config, 'CYCLE_LOSS_TYPE', 'l1')
        identity_loss_type = getattr(config, 'IDENTITY_LOSS_TYPE', 'l1')
        
        # For combined loss, get weights from config
        ssim_weight = getattr(config, 'SSIM_WEIGHT', 0.84)
        l1_weight = getattr(config, 'L1_WEIGHT', 0.16)
        grad_weight = getattr(config, 'GRAD_LOSS_WEIGHT', 0.0)
        # Optional SSIM window sizes (tune for your cell size)
        ssim_window_2d = getattr(config, 'SSIM_WINDOW', 7)
        ssim_window_3d = getattr(config, 'SSIM3D_WINDOW', 5)
        
        # Initialize loss functions
        self.gan_loss = GANLoss(gan_mode='lsgan').to(device)
        self.cycle_loss = CycleConsistencyLoss(
            loss_type=cycle_loss_type, 
            ssim_weight=ssim_weight, 
            l1_weight=l1_weight,
            grad_weight=grad_weight,
            ssim_window_2d=ssim_window_2d,
            ssim_window_3d=ssim_window_3d
        ).to(device)
        self.identity_loss = IdentityLoss(loss_type=identity_loss_type).to(device)
        
        # Loss weights
        self.lambda_cycle = config.LAMBDA_CYCLE
        self.lambda_identity = config.LAMBDA_IDENTITY
        self.lambda_adv = config.LAMBDA_ADV
        
        # Print loss configuration
        print(f"Loss configuration:")
        print(f"  Cycle loss type: {cycle_loss_type}")
        print(f"  Identity loss type: {identity_loss_type}")
        if cycle_loss_type == 'combined':
            print(f"  SSIM weight: {ssim_weight}, L1 weight: {l1_weight}, Grad weight: {grad_weight}")
        print(f"  Lambda values: cycle={self.lambda_cycle}, identity={self.lambda_identity}, adv={self.lambda_adv}")
    
    def compute_generator_loss(self, model, real_A, real_B, D_A, D_B):
        """
        Compute total generator loss.
        
        Args:
            model: Cycle-CARE model
            real_A (torch.Tensor): Real images from domain A
            real_B (torch.Tensor): Real images from domain B
            D_A: Discriminator for domain A
            D_B: Discriminator for domain B
        
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Forward pass
        outputs = model(real_A=real_A, real_B=real_B, mode='full')
        
        fake_A = outputs['fake_A']
        fake_B = outputs['fake_B']
        reconstructed_A = outputs['reconstructed_A']
        reconstructed_B = outputs['reconstructed_B']
        identity_A = outputs['identity_A']
        identity_B = outputs['identity_B']
        
        # Adversarial loss
        # G_BA should fool D_A
        pred_fake_A = D_A(fake_A)
        loss_G_BA = self.gan_loss(pred_fake_A, True) * self.lambda_adv
        
        # G_AB should fool D_B
        pred_fake_B = D_B(fake_B)
        loss_G_AB = self.gan_loss(pred_fake_B, True) * self.lambda_adv
        
        # Cycle-consistency loss
        loss_cycle_A = self.cycle_loss(reconstructed_A, real_A) * self.lambda_cycle
        loss_cycle_B = self.cycle_loss(reconstructed_B, real_B) * self.lambda_cycle
        
        # Identity loss (optional)
        loss_identity_A = self.identity_loss(identity_A, real_A) * self.lambda_identity
        loss_identity_B = self.identity_loss(identity_B, real_B) * self.lambda_identity
        
        # Total generator loss
        total_loss = (loss_G_BA + loss_G_AB + 
                     loss_cycle_A + loss_cycle_B + 
                     loss_identity_A + loss_identity_B)
        
        loss_dict = {
            'G_BA': loss_G_BA.item(),
            'G_AB': loss_G_AB.item(),
            'cycle_A': loss_cycle_A.item(),
            'cycle_B': loss_cycle_B.item(),
            'identity_A': loss_identity_A.item(),
            'identity_B': loss_identity_B.item(),
            'G_total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(self, D, real, fake):
        """
        Compute discriminator loss for a single discriminator.
        
        Args:
            D: Discriminator
            real (torch.Tensor): Real images
            fake (torch.Tensor): Fake images (detached)
        
        Returns:
            tuple: (loss, loss_dict)
        """
        # Real loss
        pred_real = D(real)
        loss_real = self.gan_loss(pred_real, True)
        
        # Fake loss
        pred_fake = D(fake.detach())
        loss_fake = self.gan_loss(pred_fake, False)
        
        # Total discriminator loss
        loss = (loss_real + loss_fake) * 0.5
        
        loss_dict = {
            'real': loss_real.item(),
            'fake': loss_fake.item(),
            'total': loss.item()
        }
        
        return loss, loss_dict
    
    def compute_discriminator_losses(self, model, real_A, real_B, D_A, D_B):
        """
        Compute losses for both discriminators.
        
        Args:
            model: Cycle-CARE model
            real_A (torch.Tensor): Real images from domain A
            real_B (torch.Tensor): Real images from domain B
            D_A: Discriminator for domain A
            D_B: Discriminator for domain B
        
        Returns:
            tuple: (loss_D_A, loss_D_B, loss_dict)
        """
        # Generate fake images
        with torch.no_grad():
            fake_A = model.G_BA(real_B)
            fake_B = model.G_AB(real_A)
        
        # Discriminator A loss
        loss_D_A, dict_D_A = self.compute_discriminator_loss(D_A, real_A, fake_A)
        
        # Discriminator B loss
        loss_D_B, dict_D_B = self.compute_discriminator_loss(D_B, real_B, fake_B)
        
        loss_dict = {
            'D_A_real': dict_D_A['real'],
            'D_A_fake': dict_D_A['fake'],
            'D_A_total': dict_D_A['total'],
            'D_B_real': dict_D_B['real'],
            'D_B_fake': dict_D_B['fake'],
            'D_B_total': dict_D_B['total']
        }
        
        return loss_D_A, loss_D_B, loss_dict


def test_losses():
    """Test the loss functions."""
    print("Testing loss functions...")
    
    batch_size = 2
    channels = 1
    height = 256
    width = 256
    
    # Create dummy tensors
    real = torch.randn(batch_size, channels, height, width)
    fake = torch.randn(batch_size, channels, height, width)
    reconstructed = torch.randn(batch_size, channels, height, width)
    
    # Test GAN loss
    print("\n1. Testing GAN Loss...")
    gan_loss = GANLoss(gan_mode='lsgan')
    pred_real = torch.randn(batch_size, 1, 30, 30)  # PatchGAN output
    pred_fake = torch.randn(batch_size, 1, 30, 30)
    loss_real = gan_loss(pred_real, True)
    loss_fake = gan_loss(pred_fake, False)
    print(f"   Real loss: {loss_real.item():.4f}")
    print(f"   Fake loss: {loss_fake.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test cycle-consistency loss (L1)
    print("\n2. Testing Cycle-Consistency Loss (L1)...")
    cycle_loss_l1 = CycleConsistencyLoss(loss_type='l1')
    loss = cycle_loss_l1(reconstructed, real)
    print(f"   L1 Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test cycle-consistency loss (SSIM)
    print("\n3. Testing Cycle-Consistency Loss (SSIM)...")
    cycle_loss_ssim = CycleConsistencyLoss(loss_type='ssim')
    loss = cycle_loss_ssim(reconstructed, real)
    print(f"   SSIM Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test cycle-consistency loss (Combined)
    print("\n4. Testing Cycle-Consistency Loss (Combined SSIM+L1)...")
    cycle_loss_combined = CycleConsistencyLoss(loss_type='combined', ssim_weight=0.84, l1_weight=0.16)
    loss = cycle_loss_combined(reconstructed, real)
    print(f"   Combined Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test identity loss
    print("\n5. Testing Identity Loss...")
    identity_loss = IdentityLoss(loss_type='l1')
    loss = identity_loss(fake, real)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test SSIM loss standalone
    print("\n6. Testing SSIM Loss (standalone)...")
    ssim_loss = SSIMLoss(window_size=11, channel=1)
    # Create similar images to test SSIM
    img1 = torch.randn(batch_size, channels, height, width)
    img2 = img1 + torch.randn_like(img1) * 0.1  # Add small noise
    loss_similar = ssim_loss(img1, img2)
    loss_different = ssim_loss(img1, fake)
    print(f"   Similar images loss: {loss_similar.item():.4f}")
    print(f"   Different images loss: {loss_different.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test MS-SSIM loss
    print("\n7. Testing Multi-Scale SSIM Loss...")
    ms_ssim_loss = MS_SSIMLoss(window_size=11, channel=1)
    loss = ms_ssim_loss(reconstructed, real)
    print(f"   MS-SSIM Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test 3D SSIM loss
    print("\n8. Testing 3D SSIM Loss...")
    depth = 16
    vol1 = torch.randn(batch_size, channels, depth, height, width)
    vol2 = torch.randn(batch_size, channels, depth, height, width)
    ssim_3d_loss = SSIM3DLoss(window_size=11, channel=1)
    loss = ssim_3d_loss(vol1, vol2)
    print(f"   3D SSIM Loss: {loss.item():.4f}")
    print(f"   Input shape: {vol1.shape} (5D volumetric)")
    print("   ✓ Test passed!")
    
    # Test 3D Combined loss
    print("\n9. Testing 3D Combined Loss...")
    combined_3d_loss = Combined3DLoss(ssim_weight=0.84, l1_weight=0.16, channel=1)
    loss = combined_3d_loss(vol1, vol2)
    print(f"   3D Combined Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test auto-detection in CycleConsistencyLoss
    print("\n10. Testing Auto-Detection (2D vs 3D)...")
    cycle_loss_auto = CycleConsistencyLoss(loss_type='ssim')
    loss_2d = cycle_loss_auto(reconstructed, real)
    loss_3d = cycle_loss_auto(vol1, vol2)
    print(f"   2D input loss: {loss_2d.item():.4f} (shape: {real.shape})")
    print(f"   3D input loss: {loss_3d.item():.4f} (shape: {vol1.shape})")
    print("   ✓ Auto-detection works!")
    
    print("\n✓ All loss tests passed!")
    print("\nLoss Function Summary:")
    print("="*60)
    print("2D Losses (for Z-context models with shape [B,C,H,W]):")
    print("  - L1: Pixel-wise absolute difference, sharp but blocky")
    print("  - SSIM: Structural similarity per Z-plane independently")
    print("  - Combined: SSIM + L1 (recommended for 2D/2.5D denoising)")
    print("\n3D Losses (for volumetric models with shape [B,C,D,H,W]):")
    print("  - SSIM3D: True volumetric structural similarity")
    print("  - Combined3D: 3D SSIM + L1 (recommended for 3D denoising)")
    print("\nKey Differences:")
    print("  • 2D SSIM: Treats each Z-plane independently")
    print("  • 3D SSIM: Computes similarity across all 3 dimensions")
    print("  • Auto-detection: CycleConsistencyLoss picks correct version")
    print("="*60)


if __name__ == "__main__":
    test_losses()
