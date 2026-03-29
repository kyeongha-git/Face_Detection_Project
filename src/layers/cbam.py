import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), # N X C/r X 1 X 1
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False) # N X C X 1 X 1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x) # N X C X 1 X 1
        max_out = self.max_pool(x) # N X C X 1 X 1

        fc_avg_out = self.fc(avg_out) # N X C X 1 X 1
        fc_max_out = self.fc(max_out) # N X C X 1 X 1
        
        out = fc_avg_out + fc_max_out # N X C X 1 X 1
        return self.sigmoid(out) # N X C X 1 X 1


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # N X 1 X H X W
        max_out, _ = torch.max(x, dim=1, keepdim=True) # N X 1 X H X W
        x = torch.cat([avg_out, max_out], dim=1) # N X 2 X H X W
        x = self.conv1(x) # N X 1 X H X W
        return self.sigmoid(x) # N X 1 X H X W 


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

