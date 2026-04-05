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

        avg_pool_1d = avg_pool.view(x.size(0), 1, -1)

        # Adaptive 1D Convolution
        avg_out = self.conv1d(avg_pool_1d)

        out = self.sigmoid(avg_out)

        return out.view(x.size(0), -1, 1, 1)


class ECA(nn.Module):
    def __init__(self, channel=512, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels=channel)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = x * self.ca(x)
        return out
