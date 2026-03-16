from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.wider_face import WiderFaceDetection, detection_collate
from src.data.data_augment import preproc
from src.utils.config import cfg_mnet, cfg_re50
from src.loss.multibox_loss import MultiBoxLoss
from src.utils.prior_box import PriorBox
import time
import datetime
import math
from thop import profile
from thop import clever_format
from src.models.our_model import OurModel

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def calculate_flops(net):
    dummy_input = torch.randn(1, 3, img_dim, img_dim)

    macs, params = profile(net, inputs=(dummy_input,))
    macs, params = clever_format([macs, params], "%.3f")
    
    print('=====================================')
    print(f'Model Architecture FLOPs & Params')
    print(f'Input Resolution: 3 x {img_dim} x {img_dim}')
    print(f'Computational complexity (MACs): {macs}')
    print(f'Number of parameters: {params}')
    print('=====================================')
    return macs, params


cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = OurModel(cfg=cfg)
print("Printing net...")
print(net)


if __name__ == '__main__':
    # Calculate FLOPs and Params  
    calculate_flops(net)