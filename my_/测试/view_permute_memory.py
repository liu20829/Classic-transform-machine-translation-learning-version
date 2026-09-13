"""
实测 view / permute 的内存行为:拆头是"重新划分边界",不搬数据
缩小版数字:batch=2, len=3, d_model=8, heads=4 -> head_dim=2
运行: G:/software/python311/python.exe my_/测试/view_permute_memory.py
"""
import torch

N, T, D, H = 2, 3, 8, 4
D_k = D // H

x = torch.arange(N * T * D, dtype=torch.float32).view(N, T, D)
print("x shape      :", tuple(x.shape))
print("x[0,0]       :", x[0, 0].tolist(), "  <- token 0 的 8 维")
print("flat memory  :", x.flatten()[:16].tolist())

# ---- view: 128 那一维被重新看成 (H, D_k) ----
v = x.view(N, T, H, D_k)
print("\nview -> v    :", tuple(v.shape))
print("v[0,0]       :", v[0, 0].tolist(), "  <- 同一份内存看成 4 组 x 2")
print("拆出的头      :", {f"head{h}": v[0, 0, h].tolist() for h in range(H)})
print("data_ptr 相同 :", x.data_ptr() == v.data_ptr(), " (view 不复制数据)")
print("stride       :", x.stride(), "->", v.stride())

# ---- permute: 只换轴的顺序 ----
p = v.permute(0, 2, 1, 3)
print("\npermute -> p :", tuple(p.shape))
print("p[0,:,0,:]   :", p[0, :, 0, :].tolist(), "  <- 4 个头在 token 0 上的向量")
print("data_ptr 相同 :", x.data_ptr() == p.data_ptr(), " (permute 也不复制)")
print("stride       :", p.stride(), " contiguous:", p.is_contiguous())

# ---- 唯一的复制发生在需要连续内存时(attention.py:42 的 .contiguous()) ----
c = p.contiguous()
print("\ncontiguous   :", c.is_contiguous(), " data_ptr 相同:", x.data_ptr() == c.data_ptr())
print("view 回 3 维  :", tuple(c.view(N, T, D).shape))

# ---- 附加:permute 是可有可无的吗? 不换顺序会算出什么 ----
q = torch.randn(N, T, H, D_k)
k = torch.randn(N, T, H, D_k)
print("\n[不 permute] q @ k^T :", tuple((q @ k.transpose(-2, -1)).shape), " <- last two dims are (H, H)")
print("[permute 后] q @ k^T :", tuple((q.permute(0, 2, 1, 3) @ k.permute(0, 2, 1, 3).transpose(-2, -1)).shape))
