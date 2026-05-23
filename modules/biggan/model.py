"""modules/biggan/model.py — BigGAN-inspired architecture (Brock et al., 2018).

Key BigGAN innovations used here:
  - Self-Attention (SA) layer in both G and D for long-range spatial coherence
  - Spectral Normalisation on all D layers (stabilises training)
  - Conditional BatchNorm in G (class-conditional, here used for domain label)
  - Orthogonal weight initialisation
  - Hierarchical latent space (skip-z connections) — simplified here

In the WSI artifact context:
  - The domain label (clean / artifact type) conditions the generator
  - Self-attention captures global tissue structure across the tile
  - Spectral norm prevents discriminator collapse on high-res tiles
  - Best choice when training on a large, diverse artifact dataset

Simplifications vs full BigGAN:
  - Image-to-image (no random noise input) — encoder replaces noise injection
  - Conditional BN uses domain label instead of class embedding
  - Truncation trick not applied (not needed for I2I)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _ortho_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SelfAttention(nn.Module):
    """
    Non-local self-attention block (Zhang et al., 2019 — SAGAN).
    Inserts long-range spatial dependencies into feature maps.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        c = max(in_channels // 8, 1)
        self.theta = spectral_norm(nn.Conv2d(in_channels, c, 1, bias=False))
        self.phi   = spectral_norm(nn.Conv2d(in_channels, c, 1, bias=False))
        self.g     = spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, bias=False))
        self.out   = spectral_norm(nn.Conv2d(in_channels // 2, in_channels, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        theta = self.theta(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C/8)
        phi   = self.phi(x).view(B, -1, N)                       # (B, C/8, N)
        attn  = F.softmax(torch.bmm(theta, phi), dim=-1)          # (B, N, N)
        g     = self.g(x).view(B, -1, N).permute(0, 2, 1)        # (B, N, C/2)
        out   = torch.bmm(attn, g).permute(0, 2, 1).view(B, C // 2, H, W)
        return x + self.gamma * self.out(out)


class ConditionalBatchNorm(nn.Module):
    """
    Conditional BatchNorm — gamma and beta are learned per domain label.
    Allows the generator to modulate style via the domain condition.
    """
    def __init__(self, num_features: int, num_domains: int):
        super().__init__()
        self.bn    = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_domains, num_features)
        self.beta  = nn.Embedding(num_domains, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        out   = self.bn(x)
        gamma = self.gamma(domain).view(-1, out.shape[1], 1, 1)
        beta  = self.beta(domain).view(-1,  out.shape[1], 1, 1)
        return out * (1 + gamma) + beta


class BigResBlockUp(nn.Module):
    """Upsampling residual block with Conditional BN — generator building block."""
    def __init__(self, in_c: int, out_c: int, num_domains: int):
        super().__init__()
        self.cbn1  = ConditionalBatchNorm(in_c, num_domains)
        self.conv1 = spectral_norm(nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
        self.cbn2  = ConditionalBatchNorm(out_c, num_domains)
        self.conv2 = spectral_norm(nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False))
        self.skip  = spectral_norm(nn.Conv2d(in_c, out_c, 1, bias=False))

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.cbn1(x, domain))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, domain))
        h = self.conv2(h)
        # Skip connection
        s = F.interpolate(x, scale_factor=2, mode="nearest")
        s = self.skip(s)
        return h + s


class BigResBlockDown(nn.Module):
    """Downsampling residual block with Spectral Norm — discriminator building block."""
    def __init__(self, in_c: int, out_c: int, downsample: bool = True):
        super().__init__()
        self.conv1     = spectral_norm(nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
        self.conv2     = spectral_norm(nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False))
        self.skip      = spectral_norm(nn.Conv2d(in_c, out_c, 1, bias=False))
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        s = self.skip(x)
        if self.downsample:
            s = F.avg_pool2d(s, 2)
        return h + s


# ─────────────────────────────────────────────────────────────────────────────
# Generator and Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class BigGenerator(nn.Module):
    """
    BigGAN-style generator with self-attention and conditional BN.
    Input: image (3, H, W) + domain label (scalar)
    Output: translated image (3, H, W)
    """
    def __init__(self, features: int = 64, num_domains: int = 2):
        super().__init__()
        f = features
        # Encoder: plain strided convs (no BN — BigGAN G uses CBN in decoder only)
        self.enc = nn.Sequential(
            spectral_norm(nn.Conv2d(3, f,   7, 1, 3)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(f, f*2, 3, 2, 1)), nn.ReLU(True),   # /2
            spectral_norm(nn.Conv2d(f*2, f*4, 3, 2, 1)), nn.ReLU(True), # /4
        )
        # Bottleneck + attention
        self.attn     = SelfAttention(f * 4)
        # Decoder with Conditional BN
        self.up1      = BigResBlockUp(f * 4, f * 2, num_domains)
        self.up2      = BigResBlockUp(f * 2, f,     num_domains)
        self.bn_out   = nn.BatchNorm2d(f)
        self.final    = spectral_norm(nn.Conv2d(f, 3, 7, 1, 3))
        self.apply(_ortho_init)

    def forward(self, x: torch.Tensor, domain: torch.Tensor = None) -> torch.Tensor:
        if domain is None:
            domain = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = self.enc(x)
        h = self.attn(h)
        h = self.up1(h, domain)
        h = self.up2(h, domain)
        h = F.relu(self.bn_out(h))
        return torch.tanh(self.final(h))


class BigDiscriminator(nn.Module):
    """
    BigGAN-style discriminator with spectral norm and self-attention.
    """
    def __init__(self, features: int = 64):
        super().__init__()
        f = features
        self.blocks = nn.Sequential(
            BigResBlockDown(3,   f,   downsample=True),
            BigResBlockDown(f,   f*2, downsample=True),
            SelfAttention(f * 2),
            BigResBlockDown(f*2, f*4, downsample=True),
            BigResBlockDown(f*4, f*8, downsample=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = spectral_norm(nn.Linear(f * 8, 1))
        self.apply(_ortho_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self.pool(h).view(h.shape[0], -1)
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class BigGAN(nn.Module):
    """
    BigGAN for cross-domain WSI artifact transfer.
    G_AB : clean tile  → artifact tile  (conditioned on target domain label)
    G_BA : artifact tile → clean tile
    D_A  : discriminates domain A
    D_B  : discriminates domain B
    """
    def __init__(self, cfg):
        super().__init__()
        c         = cfg.BigGAN
        nd        = c.get("num_domains", 2)
        self.G_AB = BigGenerator(c.generator_features, nd)
        self.G_BA = BigGenerator(c.generator_features, nd)
        self.D_A  = BigDiscriminator(c.discriminator_features)
        self.D_B  = BigDiscriminator(c.discriminator_features)
