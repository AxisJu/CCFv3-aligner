# -*- coding: utf-8 -*-
"""
ccfv3_aligner.models.network
Dual-Channel Attention ResUNet Architecture with ASPP and Gated Skip Connections.
Strictly matches the trained checkpoint weights of 260818_deeplearning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    """Residual convolutional block with Mish activation and channel attention."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.Mish(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Mish(inplace=True)
        self.se = ChannelAttention(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out


class AttentionGate(nn.Module):
    """
    Spatial Attention Gate to suppress irrelevant background and focus on nucleus boundaries.
    """
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale macro-anatomical context."""
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Mish(inplace=True)
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Mish(inplace=True)
                ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feat_list = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        feat_list.append(gp)
        concat = torch.cat(feat_list, dim=1)
        return self.project(concat)


class AttentionResUNet(nn.Module):
    """
    Dual-Channel Attention ResUNet for Multi-Class Brain Region Segmentation.
    """
    def __init__(self, in_channels=2, num_classes=16, base_c=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder stages
        self.enc1 = ResidualBlock(in_channels, base_c)          # Level 1: (B, 32, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)                         # -> (H/2, W/2)

        self.enc2 = ResidualBlock(base_c, base_c * 2)           # Level 2: (B, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2, 2)                         # -> (H/4, W/4)

        self.enc3 = ResidualBlock(base_c * 2, base_c * 4)       # Level 3: (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2, 2)                         # -> (H/8, W/8)

        self.enc4 = ResidualBlock(base_c * 4, base_c * 8)       # Level 4: (B, 256, H/8, W/8)
        self.pool4 = nn.MaxPool2d(2, 2)                         # -> (H/16, W/16)

        # Bottleneck with ASPP
        self.bottleneck = ASPP(base_c * 8, base_c * 16)         # Level 5: (B, 512, H/16, W/16)

        # Decoder stages with Attention Gates
        self.ag4 = AttentionGate(f_g=base_c * 16, f_l=base_c * 8, f_int=base_c * 4)
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_c * 16, base_c * 8)

        self.ag3 = AttentionGate(f_g=base_c * 8, f_l=base_c * 4, f_int=base_c * 2)
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_c * 8, base_c * 4)

        self.ag2 = AttentionGate(f_g=base_c * 4, f_l=base_c * 2, f_int=base_c)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_c * 4, base_c * 2)

        self.ag1 = AttentionGate(f_g=base_c * 2, f_l=base_c, f_int=base_c // 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_c * 2, base_c)

        # Final Classification Head matching checkpoint structure
        self.head = nn.Sequential(
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.Mish(inplace=True),
            nn.Conv2d(base_c, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with Attention Gates
        d4_up = self.up4(b)
        e4_att = self.ag4(g=b, x=e4)
        d4 = self.dec4(torch.cat([d4_up, e4_att], dim=1))

        d3_up = self.up3(d4)
        e3_att = self.ag3(g=d4, x=e3)
        d3 = self.dec3(torch.cat([d3_up, e3_att], dim=1))

        d2_up = self.up2(d3)
        e2_att = self.ag2(g=d3, x=e2)
        d2 = self.dec2(torch.cat([d2_up, e2_att], dim=1))

        d1_up = self.up1(d2)
        e1_att = self.ag1(g=d2, x=e1)
        d1 = self.dec1(torch.cat([d1_up, e1_att], dim=1))

        out = self.head(d1)
        return out
