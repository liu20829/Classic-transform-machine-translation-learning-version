"""
中英翻译版配置（WMT zh-en）。
与 utils.py 的区别只有三处：数据源从两个 txt 变成一个 CSV、新增限流与词表裁剪参数、
输出目录独立（避免覆盖捷克语流程的产物）。其余参数一律沿用原版。
"""

import argparse
import os
import torch

# 训练耗时统计
def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directories for local files
# 原版是两个纯文本文件（src/tgt 各一个），中英语料是单文件 CSV，两列分别是中文和英文
data_path = "my_data/WMT_dataset/wmt_zh_en_training_corpus.csv"

# 输出目录加 _ch_en 后缀：否则会覆盖原版捷克语流程的 best_model.pt 和字典 pkl
saved_chkpt = 'saved_chkpt_ch_en/'
if not os.path.exists(saved_chkpt):
  os.mkdir(saved_chkpt)

results = 'results_ch_en/'
if not os.path.exists(results):
  os.mkdir(results)

# Argument parser
# For data files
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data', type=str, default=data_path,
                    help='location of the zh-en csv data')

# 全量语料是 2500 万句对（6.36 GB），一次读进内存会爆，这里限流
parser.add_argument('--max_pairs', type=int, default=200000,
                    help='只用前 N 个句对；0 表示不限制')
# 出现次数少于该值的词不单独占编号，统一映到 UNK
parser.add_argument('--min_count', type=int, default=2,
                    help='词表裁剪阈值：出现次数 < min_count 的词归入 UNK')

# model parameter setting
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--d_model', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--n_layers', type=int, default=7,
                    help='number of enc/dec layers in each block')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of en/dec blocks')
parser.add_argument('--ffn_hidden', type=int, default=1024,
                    help='number of hidden units in FFN')
parser.add_argument('--dropout', type=float, default=0.15,
                    help='dropout probability')
parser.add_argument('--max_sent_len', type=int, default=50,
                    help='Maximum length of sentence to use for train/valid/test')

# optimizer parameter setting
parser.add_argument('--init_lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--scheduler_factor', type=float, default=0.9,
                    help='Factor with which LR will decreasing using scheduler (new_lr = old_lr * optim_factor)')
parser.add_argument('--optim_adam_eps', type=float, default=5e-9,
                    help='Adam epsilon')
parser.add_argument('--optim_patience', type=int, default=8,
                    help='Number of epochs optimizer will wait before decreasing LR')
parser.add_argument('--optim_warmup', type=int, default=16000,
                    help='Optimizer warmup')
parser.add_argument('--optim_weight_decay', type=int, default=5e-4,
                    help='Weight decay factor for optimizer')

parser.add_argument('--epoch', type=int, default=250,
                    help='Number of epochs to train')
parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient clipping')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--label_smooth_eps', type=float, default=0.1,
                    help='Hyper-parameter for label smoothening')
parser.add_argument('--early_stop_patience', type=int, default=20,
                    help='Patience for Early Stopping')

args = parser.parse_args()

# data setting
max_pairs       = None if args.max_pairs == 0 else args.max_pairs
min_count       = args.min_count

# model parameter setting
batch_size      = args.batch_size
d_model         = args.d_model
n_layers        = args.n_layers
n_heads         = args.n_heads
ffn_hidden      = args.ffn_hidden
dropout         = args.dropout
max_sent_len    = args.max_sent_len

# optimizer parameter setting
init_lr = args.init_lr
factor = args.scheduler_factor
adam_eps = args.optim_adam_eps
patience = args.optim_patience
warmup = args.optim_warmup
epoch = args.epoch
clip = args.clip
weight_decay = args.optim_weight_decay
label_smooth_eps = args.label_smooth_eps
lr_mul = 2
early_stop_patience = args.early_stop_patience
inf = float('inf')


if __name__ == "__main__":
    # 配置本身没有逻辑，只核对派生值：目录建出来了、参数解析成了正确的类型
    print('device      :', device)
    print('data_path   :', data_path, '| 存在:', os.path.exists(data_path))
    print('saved_chkpt :', saved_chkpt, '| 存在:', os.path.exists(saved_chkpt))
    print('results     :', results, '| 存在:', os.path.exists(results))
    print('max_pairs   :', max_pairs, '(None = 不限制)')
    print('min_count   :', min_count)

    assert os.path.exists(data_path), '找不到中英文语料 CSV'
    assert os.path.exists(saved_chkpt) and os.path.exists(results), '输出目录没建出来'
    assert max_pairs is None or max_pairs > 0, 'max_pairs 应 > 0 或 0(表示不限制)'
    assert min_count >= 1, 'min_count 至少为 1'
    print('配置自检通过')
