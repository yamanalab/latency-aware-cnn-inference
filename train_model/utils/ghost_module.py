import math

import torch
import torch.nn as nn


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        intrinstic_channels = math.ceil(out_channels / ratio)
        ghosts_channels = intrinstic_channels * (ratio-1)

        self.primary_conv = nn.Conv2d(in_channels, intrinstic_channels, kernel_size, stride, padding)

        # Depth-wise convolution to generate ghost features
        self.linear_op = nn.Conv2d(intrinstic_channels,
                                   ghosts_channels,
                                   kernel_size=dw_size, stride=1, padding=dw_size//2,
                                   groups=intrinstic_channels, bias=False)

    def forward(self, x):
        intrinstic_features = self.primary_conv(x)
        ghost_features = self.linear_op(intrinstic_features)
        out = torch.cat([intrinstic_features, ghost_features], dim=1)
        return out[:, :self.out_channels, :, :]
