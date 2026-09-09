import torch
from torch import nn

# ########################################################################
# # POS ENCODING - Using sin/cos
# Input : Sentence length, Word embedding size, Device (CPU/GPU?)
# ########################################################################
class PositionEncoding(nn.Module):
	def __init__(self):
		super(PositionEncoding, self).__init__()

	def forward(self, seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")):
		# 位置 0..seq_len-1
		pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(-1, 1)  # (seq, 1)
		# 偶数维下标 2i: 0, 2, 4, ..., dim_model-2
		idx = torch.arange(0, dim_model, 2, dtype=torch.float, device=device)         # (dim_model/2,)
		# 论文公式: PE(pos,2i)=sin(pos/10000^(2i/d_model)), PE(pos,2i+1)=cos(同式)
		# 频率随维度增大而递减:低维高频率编码近距,高维低频率编码远距
		angles = pos / 10000 ** (idx / dim_model)    # (seq, dim_model/2)

		pe = torch.zeros(seq_len, dim_model, device=device)
		pe[:, 0::2] = torch.sin(angles)  # 偶数列 sin
		pe[:, 1::2] = torch.cos(angles)  # 奇数列 cos,与相邻偶数列共用同一频率
		return pe.unsqueeze(0)           # (1, seq, dim_model)