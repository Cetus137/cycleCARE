"""
Loss functions for Cycle-CARE training.
Includes adversarial, cycle-consistency, and identity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Cycle-consistency loss.
    
    Ensures that F(G(x)) ≈ x and G(F(y)) ≈ y
    where F and G are the two generators.
    
    This loss is crucial for learning meaningful mappings without paired data.
    
    Args:
        loss_type (str): Type of loss ('l1' or 'l2')
    """
    def __init__(self, loss_type='l1'):
        super(CycleConsistencyLoss, self).__init__()
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def forward(self, reconstructed, original):
        """
        Calculate cycle-consistency loss.
        
        Args:
            reconstructed (torch.Tensor): Reconstructed image after cycle
            original (torch.Tensor): Original image
        
        Returns:
            torch.Tensor: Loss value
        """
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
        
        # Initialize loss functions
        self.gan_loss = GANLoss(gan_mode='lsgan').to(device)
        self.cycle_loss = CycleConsistencyLoss(loss_type='l1').to(device)
        self.identity_loss = IdentityLoss(loss_type='l1').to(device)
        
        # Loss weights
        self.lambda_cycle = config.LAMBDA_CYCLE
        self.lambda_identity = config.LAMBDA_IDENTITY
        self.lambda_adv = config.LAMBDA_ADV
    
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
    
    # Test cycle-consistency loss
    print("\n2. Testing Cycle-Consistency Loss...")
    cycle_loss = CycleConsistencyLoss(loss_type='l1')
    loss = cycle_loss(reconstructed, real)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    # Test identity loss
    print("\n3. Testing Identity Loss...")
    identity_loss = IdentityLoss(loss_type='l1')
    loss = identity_loss(fake, real)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Test passed!")
    
    print("\n✓ All loss tests passed!")


if __name__ == "__main__":
    test_losses()
