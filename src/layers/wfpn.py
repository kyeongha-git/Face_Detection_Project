import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.conv_block import conv_bn, conv_bn1X1

class WFPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(WFPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.alpha_conv1 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1)
        self.alpha_conv2 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1)
        
    def forward(self, input):
        # names = list(input.keys())
        output1 = self.output1(input[0]) #input[0] = B_0, output1 = O_1
        output2 = self.output2(input[1]) #input[1] = B_1, output2 = O_2
        output3 = self.output3(input[2]) #input[2] = B_2, output3 = O_3
        
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        alpha_1 = torch.sigmoid(self.alpha_conv1(up3))
        
        output2 = self.merge1(alpha_1 * up3 + (1-alpha_1) * output2)
        
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        alpha_2 = torch.sigmoid(self.alpha_conv2(up2))
        
        output1 = self.merge2(alpha_2 * up2 + (1-alpha_2) * output1)
        
        out = [output1, output2, output3]
        return out