"""
用法：
    冒烟测试  python train_ch_en.py --max_pairs 2000 --epoch 2
    正式训练  python train_ch_en.py
"""
from torch import nn
import time,os
from torch import optim
from tqdm import tqdm
from torch.nn import functional

