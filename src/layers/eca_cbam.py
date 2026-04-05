import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


def AdaptiveKernelSize(channels):
    k = math.ceil((math.log2(channels) + 1) / 2)
    return k if k % 2 == 1 else k + 1


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        adaptive_kernel_size = AdaptiveKernelSize(channels)
        self.conv1d = nn.Conv1d(
            1,
            1,
            kernel_size=adaptive_kernel_size,
            stride=1,
            padding=adaptive_kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        avg_pool_1d = avg_pool.view(x.size(0), 1, -1)
        max_pool_1d = max_pool.view(x.size(0), 1, -1)

        # Adaptive 1D Convolution
        avg_out = self.conv1d(avg_pool_1d)
        max_out = self.conv1d(max_pool_1d)

        combined = avg_out + max_out
        out = self.sigmoid(combined)

        return out.view(x.size(0), -1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ECA_CBAM(nn.Module):
    def __init__(self, channel=512, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels=channel)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual
