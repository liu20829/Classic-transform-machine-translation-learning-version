"""
实测 scaled_dot_product_attn 各步张量形状与 pad 行的 softmax 行为
用项目真实代码跑,不靠推算。运行: G:/software/python311/python.exe my_/测试/shape_trace.py
"""
import sys
sys.path.insert(0, r"G:\PythonProject\follow-github-learn-agent\Transformer-for-Machine-Translation-main")

import torch
from model.attention import MultiHeadAttention
from model.transformer import Transformer
import tokenizer

N, T, Tk, D, H = 3, 52, 52, 256, 8   # batch=3, seq=52, d_model=256, heads=8
D_k = D // H                          # head_dim = 32


# ---- 0. 分词后真实长度(公式: 1 SOS + L 词 + 1 EOS + (50-L) pad = 52) ----
class _Stub:
    word2index = {w: i + 3 for i, w in enumerate("a b c".split())}


print("tokenize len        :", len(tokenizer.tokenize("a b c", _Stub)))

# ---- 1. 自注意力: q/k/v 同源 ----
mha = MultiHeadAttention(D, H, 0.0, 'cpu').eval()
tf = Transformer(None, None, 'cpu', 0)          # 只为调用 make_input_mask

ids = torch.randint(3, 100, (N, T))
ids[:, 10:] = 0                                 # 后 42 个位置是 pad(PAD_TOKEN=0)
src = torch.randn(N, T, D)

mask = tf.make_input_mask(ids)
print("ids/src             :", tuple(ids.shape), tuple(src.shape))
print("mask                :", tuple(mask.shape), "<- 只按 key 位置屏蔽并广播")

q = mha.q(src).view(N, -1, H, D_k).permute(0, 2, 1, 3)
k = mha.k(src).view(N, -1, H, D_k).permute(0, 2, 1, 3)
v = mha.v(src).view(N, -1, H, D_k).permute(0, 2, 1, 3)
print("q/k/v              :", tuple(q.shape), "= [N, n_heads, len, head_dim]")

scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / D ** 0.5
print("scores             :", tuple(scores.shape), "= [N, n_heads, len_q, len_k]")

masked = scores.masked_fill(mask == 0, -1e10)
print("被整行屏蔽的行数     :", int((masked[0] == -1e10).all(-1).sum()),
      "(=0 说明没有 query 会看到全 -1e10 的一行)")

weights = torch.softmax(masked, dim=-1)
out = torch.matmul(weights, v)
print("weights            :", tuple(weights.shape))
print("out                :", tuple(out.shape))
print("merged(多头拼接)    :", tuple(out.permute(0, 2, 1, 3).contiguous().view(N, T, D).shape))

# ---- 2. pad 位置当 query 时到底算出什么 ----
pad_q = weights[0, 0, 20]      # 位置 20 是 pad,拿它当 query
real_q = weights[0, 0, 5]      # 位置 5 是真实词
print("pad  query 行 权重和 : 真词位=%.4f  pad位=%.4f  行内std=%.2e"
      % (pad_q[:10].sum().item(), pad_q[10:].sum().item(), pad_q.std().item()))
print("real query 行 权重和 : 真词位=%.4f  pad位=%.4f  行内std=%.2e"
      % (real_q[:10].sum().item(), real_q[10:].sum().item(), real_q.std().item()))

# ---- 3. 交叉注意力: q 来自解码器, k/v 来自编码器, 两个长度不同 ----
q2 = torch.randn(N, H, 7, D_k)
k2 = torch.randn(N, H, Tk, D_k)
v2 = torch.randn(N, H, Tk, D_k)
print("cross scores       :", tuple(torch.matmul(q2, k2.permute(0, 1, 3, 2)).shape),
      "= [N, n_heads, len_tgt, len_src]")

# ---- 4. 走一遍真实 forward ----
attn, w = mha(src, src, src, mask)
print("mha.forward 输出    :", tuple(attn.shape), tuple(w.shape))
