import torch
from torch import nn
# embedding_positional 直接复用编码器那份:逻辑完全一样,只是词表换成目标语言的
# (机器翻译要另建词表;如果是不改变语言的预测,比如对话,可以复用同一个)
from transformer_encoder import MultiHeadAttention,feedforwardlayer,embedding_positional


class transformer_decoder_layer(nn.Module):
    def __init__(self,model_dim,num_heads,ff_size,device,dropout):
        super().__init__()
        self.masked_multi_head_attention=MultiHeadAttention(model_dim,num_heads,dropout,device)
        self.norm1=nn.LayerNorm(model_dim)
        self.multi_head_attention=MultiHeadAttention(model_dim,num_heads,dropout,device)
        self.norm2=nn.LayerNorm(model_dim)
        self.feed_forward=feedforwardlayer(model_dim,ff_size,dropout)
        self.norm3=nn.LayerNorm(model_dim)
        self.dropout=nn.Dropout(dropout)

        pass
    def forward(self,target,encoder_out,target_mask,encoder_out_mask):
        """
        target_mask:自注意力用的掩码,是 padding掩码 与 下三角因果掩码 的与
                    光挡PAD不够,每个位置还必须挡住它右边的未来位置
        encoder_out_mask :交叉注意力用的掩码,挡的是源句的padding(源句整句可见,没有因果问题)
        """
        # print(f"target.shape\n{target.shape}")
        # print(f"target_mask.shape\n{target_mask.shape}")
        y=self.masked_multi_head_attention(target,target,target,target_mask)
        y1=self.norm1(self.dropout(y)+target)

        y2=self.multi_head_attention(y1,encoder_out,encoder_out,encoder_out_mask)
        y3=self.norm2(self.dropout(y2)+y1)

        y4=self.feed_forward(y3)
        y5=self.norm3(self.dropout(y4)+y3)
        return y5
        pass
    pass

class transformer_decoder(nn.Module):
    def __init__(self,y_vocab_size,len_q,model_dim,num_heads,ff_size,decoder_num,device,dropout):
        super().__init__()
        # self.transformer_decoder_layer=transformer_decoder_layer(model_dim,num_heads,ff_size,device,dropout)
        self.embedding_positional=embedding_positional(y_vocab_size,model_dim,len_q,device,dropout)

        # self.dropout=nn.Dropout(dropout)

        # decoder_num_ls=nn.ModuleList()

        self.transformer_decoder_num=nn.ModuleList(
            [transformer_decoder_layer(model_dim,num_heads,ff_size,device,dropout)
                for decoder in range(decoder_num) ]
        )

        self.linear=nn.Linear(model_dim,y_vocab_size)

        pass
    def forward(self,target,encoder_out,target_mask,encoder_out_mask):
        y=self.embedding_positional(target)
        # y=self.dropout(y)
        for decoder in self.transformer_decoder_num:
            y=decoder(y,encoder_out,target_mask,encoder_out_mask)
            pass
        y=self.linear(y)
        return y
        pass
    pass


if __name__=="__main__":
    # ---- 虚拟输入配置 ----
    # 这里直接测解码器层,输入是已经过 embedding 的向量,所以没有词表,形状里是 model_dim
    batch_size = 2
    len_q      = 6      # 解码器输入长度
    len_src    = 8      # 源句长度
    model_dim  = 32
    num_heads  = 4
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    target      = torch.randn(batch_size, len_q,   model_dim, device=device)   # 模拟解码器输入向量
    encoder_out = torch.randn(batch_size, len_src, model_dim, device=device)   # 模拟编码器输出

    # 自注意力掩码:全可见 与 下三角 的与
    # 故意不挡 PAD,这样只剩因果在起作用,能干净地测出它
    target_pad_mask  = torch.ones(batch_size, 1, 1, len_q, dtype=torch.bool, device=device)
    target_tril_mask = torch.tril(torch.ones(len_q, len_q, device=device)).bool()
    target_mask      = target_pad_mask & target_tril_mask

    # 交叉注意力掩码:源句后 3 位当作 PAD 挡掉
    input_mask = torch.ones(batch_size, 1, 1, len_src, dtype=torch.bool, device=device)
    input_mask[:, :, :, 5:] = False

    layer = transformer_decoder_layer(model_dim, num_heads, model_dim * 4, device, 0.0).to(device)
    layer.eval()

    print('target      :', tuple(target.shape))
    print('encoder_out :', tuple(encoder_out.shape))
    print('target_mask :', tuple(target_mask.shape))
    print('input_mask  :', tuple(input_mask.shape))

    with torch.no_grad():
        out = layer(target, encoder_out, target_mask, input_mask)

    print('输出 out    :', tuple(out.shape))
    print('有 NaN 吗   :', bool(torch.isnan(out).any()))

    # 校验:形状必须保持 [batch, len_q, model_dim](层不改变形状),且没有 NaN
    assert out.shape == (batch_size, len_q, model_dim), out.shape
    assert not torch.isnan(out).any(), '输出出现 NaN'
    print('smoke test 通过')

    # ---- 验证因果掩码真的生效 ----
    # 只改最后一个位置的输入,它左边的位置看不到它,输出应当纹丝不动
    target2 = target.clone()
    target2[:, -1] = target2[:, -1] + 1.0

    with torch.no_grad():
        out2 = layer(target2, encoder_out, target_mask, input_mask)

    diff_past = (out[:, :-1] - out2[:, :-1]).abs().max().item()   # 前序位置:应约为 0
    diff_last = (out[:, -1] - out2[:, -1]).abs().max().item()     # 被改的位置自己:应大于 0

    # 对照组:把因果那一半去掉(掩码全开),前序位置立刻被未来的输入污染
    no_causal_mask = torch.ones(batch_size, 1, len_q, len_q, dtype=torch.bool, device=device)
    with torch.no_grad():
        out3 = layer(target2, encoder_out, no_causal_mask, input_mask)
    diff_nocausal = (out[:, :-1] - out3[:, :-1]).abs().max().item()

    print('前序位置输出差异          :', diff_past, '(应约为 0)')
    print('末位输出差异              :', diff_last, '(应大于 0)')
    print('对照组(无因果掩码)前序差异:', diff_nocausal, '(应大于 0)')

    assert diff_past < 1e-6, '因果掩码没生效:位置 t 能看到位置 t+1 的输入'
    assert diff_last > 0, '改动的输入没传导到输出'
    assert diff_nocausal > 0, '对照组也没变化,说明这次改动本身没传导到输出'
    print('因果掩码验证通过')
