import torch
import torch.nn as nn
from thop import profile, clever_format
import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.our_model import OurModel
from src.utils.config import cfg_mnet
from src.layers.conv_block import DeformableConv2d

# Custom hook for DeformableConv2d to include the core convolution FLOPs
def count_deform_conv(m, x, y):
    kernel_ops = m.regular_conv.weight.size()[2:].numel()
    in_channels = m.regular_conv.in_channels
    out_channels = m.regular_conv.out_channels
    groups = m.regular_conv.groups
    
    h_out, w_out = y.shape[2:]
    # MACs = H_out * W_out * C_out * (C_in / groups) * K * K
    total_ops = h_out * w_out * out_channels * (in_channels // groups) * kernel_ops
    
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def main():
    parser = argparse.ArgumentParser(description='Calculate Model FLOPs & Parameters')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    parser.add_argument('--attention_type', type=str, default='eca_cbam', 
                        choices=['none', 'eca', 'cbam', 'eca_cbam'], help='Attention type (for our_model)')
    args = parser.parse_args()

    # Create model
    # Note: OurModel in 'preivious' might not accept attention_type, so we handle it gracefully.
    try:
        model = OurModel(cfg=cfg_mnet, attention_type=args.attention_type)
    except TypeError:
        # If it doesn't accept attention_type, it's likely the 'preivious' version
        model = OurModel(cfg=cfg_mnet)
    
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_size = args.image_size
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    custom_ops = {DeformableConv2d: count_deform_conv}
    
    macs, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    
    attn_str = f" | Attention: {args.attention_type}" if hasattr(model, 'attention_type') else ""
    
    print('\n' + '='*50)
    print(f' Model Architecture FLOPs & Params')
    print(f' Input Resolution: 3 x {input_size} x {input_size}{attn_str}')
    print(f' Computational complexity (MACs): {macs}')
    print(f' Number of parameters: {params}')
    print('='*50 + '\n')

if __name__ == '__main__':
    main()