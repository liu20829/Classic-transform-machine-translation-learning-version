import torch
seq_len=10
pos = torch.arange(seq_len, dtype=torch.float).reshape(1, -1, 1)
idx = torch.arange(0, 25, 2, dtype=torch.float)
print(pos)
print(idx)