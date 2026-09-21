from torch import nn, Tensor
import torch


class feedforwardlayer(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size)
        )

    def forward(self, attention_out):
        feed_out = self.ff_layer(attention_out)
        return feed_out
        pass

    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float, device):
        super().__init__()
        """
        多头注意力层
        前馈网络层
        """
        self.token_dim = token_dim  # query的嵌入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.device = device

        self.q = nn.Linear(token_dim, token_dim)
        self.k = nn.Linear(token_dim, token_dim)
        self.v = nn.Linear(token_dim, token_dim)

        self.head_dim = token_dim // num_heads

        self.head_linear = nn.Linear(token_dim, token_dim)

        self.dropout = nn.Dropout(dropout)
        pass

    def scaled_dot_product_attn(self, q, k, v, mask):
        """
        该函数接收的 q,k,v 是经过线性变换、拆头、换轴之后的
        q/k/v.shape = [批大小, 头的数量, 序列长度, 单头维度 head_dim=token_dim//num_heads]
                    = [batch_size,num_heads,len_q,head_dim]
        mask.shape  = [批大小, 1, 1, key的序列长度]      # padding 掩码,编码器用
                    = [批大小, 1, 查询长度, key的长度]    # target 掩码,解码器自注意力用 =[batch_size, 1, len_q, len_q]

        """
        # 先说去缩放尺寸
        sqrt_dim = self.token_dim ** 0.5

        # 矩阵乘法过后tmp.shape=[batch_size,num_heads,len_q,len_q]
        # print(f"k.permute(0,1,3,2).shape{k.permute(0, 1, 3, 2).shape}")
        # print(f"q.shape{q.shape}")
        # q, k.permute(0, 1, 3, 2)此时的矩阵乘法就是计算q中单头token的特征


        tmp = torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt_dim
        # tmp=[batch_size, num_heads, len_q, len_q]

        # print(f"tmp1.shape\n{tmp.shape}")
        if mask is not None:
            tmp = tmp.masked_fill(mask == 0, -1e10)
            # print(f"tmp2.shape\n{tmp.shape}")
            pass

        softmax_out = torch.softmax(tmp, dim=-1)

        # 矩阵乘法过后=[batch_size,num_heads,len_q,head_dim]
        attention = torch.matmul(self.dropout(softmax_out), v)

        # 开始恢复attention的形状
        # [batch_size,num_heads,len_q,head_dim]->[batch_size,len_q,num_heads,head_dim]->[batch_size,len_q,token_dim]
        attention = attention.permute(0, 2, 1, 3).contiguous().view(q.shape[0], -1, self.token_dim)  # 调整数值内存位置

        return attention
        pass

    def forward(self, query, key, value, mask=None):
        # query.shape=[batch_size,len_q,token_dim]
        batch = query.shape[0]
        q = self.q(query)  # .shape=[batch_size,len_q,token_dim]
                            # 每个维度的物理意义[批大小，序列长度，无法解释（虽然维度和原始token的维度一样但是经过线性变换之后无法解释了，但是可以肯定每一行的数据代表一个token的特征信息,所以在某种意义上说还是token特征信息）]
        k = self.k(key)
        v = self.v(value)

        # 进行形状调整维缩放点积注意力做形状准备
        # [batch_size, len_q, token_dim]
        # ->[batch_size,(len_q*token_dim/num_heads/token_dim*num_head)=len_q,num_heads,(token_dim/num_heads)=head_dim]
        # =[batch_size,len_q,num_heads,head_dim]
        # ->[batch_size,num_heads,len_q,head_dim]
        q = q.view(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = self.scaled_dot_product_attn(q, k, v, mask)

        attention = self.head_linear(attention)

        return attention
        pass

    pass


class transformer_encoder_layer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, ff_size, device):
        super().__init__()
        """
        该来包含注意力编码层和前向传播编码层
        """
        self.attention_encoder = MultiHeadAttention(model_dim, num_heads, dropout, device)
        self.norm1 = nn.LayerNorm(model_dim)
        self.feedforward_encoder = feedforwardlayer(model_dim, ff_size, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, x, mask):
        attention_out = self.attention_encoder(x, x, x, mask)
        attention_norm_out = self.norm1(self.dropout(attention_out) + x)

        feedforward_out = self.feedforward_encoder(attention_norm_out)

        feedforward_norm_out = self.norm2(self.dropout(feedforward_out) + attention_norm_out)
        return feedforward_norm_out
        pass

    pass


class positional(nn.Module):
    def __init__(self, model_dim: int, len_q: int, device: str):
        super().__init__()
        self.model_dim = model_dim
        self.len_q = len_q
        self.device = device
        pass

    def forward(self, len_q=None):
        # 默认用构造时定死的长度；解码器推理时序列逐轮变长，由调用方传入实际长度
        len_q = self.len_q if len_q is None else len_q
        pos = torch.arange(0, len_q, dtype=torch.float, device=self.device).reshape(-1, 1)

        idx = torch.arange(0, self.model_dim, 2, device=self.device, dtype=torch.float)

        angles = pos / (10000 ** (idx / self.model_dim))

        pe = torch.zeros(len_q, self.model_dim, device=self.device)

        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe.unsqueeze(0)  # 【0，len_q,model_dim】
        pass


class embedding_positional(nn.Module):
    def __init__(self, src_vocab_size, model_dim, len_q, device, dropout):
        super().__init__()
        # self.model_dim=dim_model
        # self.
        self.embedding = nn.Embedding(src_vocab_size, model_dim)

        self.positional = positional(model_dim, len_q, device)
        self.coefficient = torch.sqrt(torch.FloatTensor([model_dim])).to(device)

        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, x):
        embedding_x = self.embedding(x)
        # 按 x 的实际长度取位置编码，不能写死构造长度（否则解码器推理时位置编码会广播出错误的序列长度）
        positional_x = self.positional(x.shape[1])
        embedding_x_pos = self.dropout((embedding_x * self.coefficient + positional_x))
        return embedding_x_pos
        pass

    pass


class transformer_encoder(nn.Module):
    """
    该类包含原始token输入后要先embedding->post位置编码->多头注意力机制->前馈网络
    """

    def __init__(self,
                 src_vocab_size,  # 嵌入矩阵大小
                 model_dim,  # 每个toekn的维度，
                 len_q,  # 序列长度
                 num_heads: int = 8,  # 注意力头
                 feedforward_dim: int = 2048,  # 前馈网络维度
                 dropout: float = 0.1,
                 device='cuda',
                 encode_num: int = 6
                 ):
        super().__init__()
        self.embedding_positional = embedding_positional(src_vocab_size, model_dim, len_q, device, dropout)
        # self.transformer_layer=transformer_encoder_layer(model_dim,num_heads,dropout,feedforward_dim,device)
        self.encode_num_layer = nn.ModuleList()
        for i in range(encode_num):
            self.encode_num_layer.append(
                transformer_encoder_layer(model_dim, num_heads, dropout, feedforward_dim, device))

        pass

    def forward(self, x, mask):
        # mask掩码使遮蔽填充的元素的
        x = self.embedding_positional(x)
        # x=self.encode_num_layer(x)
        for transformer_encoder_layer_i in self.encode_num_layer:
            x = transformer_encoder_layer_i(x, mask)
            pass
        return x

        pass

    pass


if __name__ == "__main__":
    # ---- 虚拟输入配置 ----
    batch_size = 2  # 批大小
    len_q = 10  # 序列长度(必须和构造时的 len_q 一致,位置编码长度在构造时就定死了)
    model_dim = 32  # 每个 token 的维度
    num_heads = 4  # 注意力头数(model_dim 必须能被整除)
    vocab_size = 100  # 词表大小
    pad_idx = 0  # 填充位编号
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机 token 编号模拟两句话,第 8 位之后全部填 pad(模拟变长句子的填充)
    x = torch.randint(3, vocab_size, (batch_size, len_q), device=device)
    x[:, 8:] = pad_idx

    # padding mask:形状 [batch, 1, 1, len_q],0 表示屏蔽(和参考实现约定一致)
    mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)

    model = transformer_encoder(
        vocab_size,  # src_vocab_size
        model_dim,
        len_q,
        num_heads,
        model_dim * 4,  # feedforward_dim
        0.0,  # dropout 关掉,保证结果可复现
        device,
        2,  # 编码器层数少一点,跑得快
    ).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x, mask)

    print('输入 x    :', tuple(x.shape))
    print('mask      :', tuple(mask.shape))
    print('输出 out  :', tuple(out.shape))
    print('有 NaN 吗 :', bool(torch.isnan(out).any()))

    # 校验:形状必须回到 [batch, len_q, model_dim],且没有 NaN
    assert out.shape == (batch_size, len_q, model_dim), out.shape
    assert not torch.isnan(out).any(), '输出出现 NaN'
    print('smoke test 通过')

    # ---- 验证 mask 真的生效 ----
    # 把 pad 位置的 token 编号改掉再跑一遍(复用原来的 mask:pad 的判定依据是原句子长度,不是改后的编号)
    x2 = x.clone()
    x2[:, 8:] = pad_idx + 5
    with torch.no_grad():
        out2 = model(x2, mask)

    # pad 作为 key 已被屏蔽,它的 value 不该参与加权求和 -> 非 pad 位置的输出应当不变
    diff_real = (out[:, :8] - out2[:, :8]).abs().max().item()
    # pad 位置自己当 query 时,查询向量来自自己的 embedding,输出本来就该变
    diff_pad = (out[:, 8:] - out2[:, 8:]).abs().max().item()

    # 对照组:不给 mask,同样的改动就会传导到真实位置 -> 证明上面的不变性确实来自 mask
    with torch.no_grad():
        out3 = model(x2, None)
    diff_nomask = (out[:, :8] - out3[:, :8]).abs().max().item()

    print('非 pad 位输出差异        :', diff_real, '(应约为 0)')
    print('pad   位输出差异        :', diff_pad, '(应大于 0)')
    print('对照组(无 mask)非 pad 差异:', diff_nomask, '(应大于 0)')

    assert diff_real < 1e-6, 'mask 没生效:pad 位影响了真实位置的输出'
    assert diff_pad > 0, 'pad 位输出没变,pad 位的输入没改成功?'
    assert diff_nomask > 0, '对照组也没变化,说明这次改动本身没传导到输出'
    print('mask 验证通过')
