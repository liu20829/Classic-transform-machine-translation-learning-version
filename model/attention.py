"""
@author : Shashank Agarwal
@when : 03-12-2021
@homepage : https://github.com/shashankag14
"""

import torch
from torch import Tensor
from torch import nn

class MultiHeadAttention(nn.Module):
	def __init__(self, dim_model: int, num_heads: int, dropout: float, device):
		super().__init__()
		self.device = device
		self.dim_model = dim_model
		self.num_heads = num_heads

		self.q = nn.Linear(dim_model, dim_model)
		self.k = nn.Linear(dim_model, dim_model)
		self.v = nn.Linear(dim_model, dim_model)

		self.head_dim = dim_model // num_heads
		self.linear = nn.Linear(dim_model, dim_model)

		self.dropout = nn.Dropout(dropout)

	def scaled_dot_product_attn(self, q, k, v, mask):
		N = q.shape[0]
		# q / k / v  = [N, H, len, D_k]，N是批大小，H注意力头数量，
		# mask  =  [N, 1, 1, len_k]


		# temp -> [N, num_headus, len_q, len_k]
		scale = (self.dim_model ** 0.5)

		# 矩阵乘法过后形状变为[N, H, len, len]
		temp = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale

		if mask is not None:
			temp = temp.masked_fill(mask == 0, -1e10)#输入的句子为了对齐长度进行了占位符填充，这一步的遮蔽就是把填充的给遮蔽了，经过softmax转换之后约等于0

		softmax_out = torch.softmax(temp, dim=-1)
		# matmul faster than einsum
		# Apply dropout on softmax and then multiply with value
		attention = torch.matmul(self.dropout(softmax_out), v)
		# 4d - [N, len_q, num_head, head_dim]
		attention = attention.permute(0, 2, 1, 3).contiguous()#contiguous()`  **返回张量在内存中连续存储的副本
		# 3d - [N, len_q, num_head*head_dim]
		attention = attention.view(N, -1, self.dim_model)
		return attention, softmax_out

	def forward(self, query, key, value, mask=None):
		N = query.shape[0]
		# query的形状【N批大小，q_len句子长度，dim每个token的维度大小】

		query = self.q(query)
		key = self.k(key)
		value = self.v(value)     

		# x (k/q/v) =query的形状【N批大小，q_len句子长度，dim每个token的维度大小】->
		# (N,q_len*dim/dim,8,dim/8)=(N,len_x,num_heads,head_dim)->(N, num_heads, len_x, head_dim)
		query = query.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		key = key.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
		value = value.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		# [N, H, len, D_k]
		attention, softmax_out = self.scaled_dot_product_attn(query, key, value, mask)

		# [N, H, len, D_k]
		attention = self.linear(attention)#这一步是为了将多头注意力的数据进行线性融合

		return attention, softmax_out
