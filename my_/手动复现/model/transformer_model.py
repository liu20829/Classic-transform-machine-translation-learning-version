import torch
from torch import nn
from transformer_decoder import transformer_decoder
from transformer_encoder import transformer_encoder
class transformer_model(nn.Module):
    def __init__(self,src_vocab_size,model_dim,num_heads,
                 y_vocab_size,ff_size,decoder_num,len_q,
                 feedforward,dropout,padding_index,device,encode_num):
        super().__init__()
        self.encoder=transformer_encoder(src_vocab_size,model_dim,len_q,num_heads,feedforward,dropout,device,encode_num)
        self.decoder=transformer_decoder(y_vocab_size,len_q,model_dim,num_heads,ff_size,decoder_num,device,dropout)
        self.padding_index=padding_index
        self.device=device
        pass
    pass
    def make_input_mask(self,input):
        # 输入的input的形状是（batch_size,len_q）
        # 计算不等于后得到的是只要是不等于padding_index为1，等于的为0得到一个对pad掩码的矩阵
        # (batch_size,len_q)->(batch_size,1,len_q)->(batch_size,1,1,len_q)
        input_mask=(input!=self.padding_index).unsqueeze(1).unsqueeze(2)
        return input_mask
        pass
    def make_target_mask(self,target):
        target_pad_mask=(target!=self.padding_index).unsqueeze(1).unsqueeze(2)

        # 生成倒三角掩码
        target_tril_mask=torch.tril(torch.ones((target.shape[1],target.shape[1]),device=self.device)).bool()

        return target_pad_mask & target_tril_mask
        pass
    def forward(self,src,tgt):
        # 1运行编码器
        src_mask=self.make_input_mask(src)
        encoded_input =self.encoder(src,src_mask)

        # 2.运行解码器
        target_mask=self.make_target_mask(tgt)
        output=self.decoder(tgt,encoded_input,target_mask,src_mask)
        return output

        pass


if __name__=="__main__":
    # ---- 虚拟输入配置 ----
    # 形状够跑通就行,数字取小值让 CPU 也能秒出结果
    batch_size, len_src, len_tgt = 2, 8, 6
    src_vocab, y_vocab, model_dim = 20, 20, 32
    num_heads, ff_size = 4, 128
    encode_num, decoder_num, len_q = 2, 2, 50
    padding_index, dropout = 0, 0.0     # dropout 关掉,免得随机性干扰断言
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model=transformer_model(src_vocab,model_dim,num_heads,
                 y_vocab,ff_size,decoder_num,len_q,
                 ff_size,dropout,padding_index,device,encode_num).to(device)
    model.eval()

    # token 从 4 起取,避开 0/1/2/3(分别是 PAD/SOS/EOS/UNK),免得随机值混进掩码断言
    src=torch.randint(4,src_vocab,(batch_size,len_src),device=device)
    src[:,5:]=padding_index             # 源句后 3 位当 PAD
    tgt=torch.randint(4,y_vocab,(batch_size,len_tgt),device=device)
    tgt[:,4:]=padding_index             # 目标句后 2 位当 PAD

    # ---- 1. 两个掩码的形状 ----
    # 源句掩码:每个 query 看到的都是整句,只挡 PAD,所以 query 那一维是 1
    # 目标掩码:query 能看到多远随位置变,所以 query 那一维是完整的 len_tgt
    src_mask=model.make_input_mask(src)
    tgt_mask=model.make_target_mask(tgt)
    assert src_mask.shape==(batch_size,1,1,len_src),src_mask.shape
    assert tgt_mask.shape==(batch_size,1,len_tgt,len_tgt),tgt_mask.shape

    # ---- 2. PAD 位置被标成 False ----
    assert src_mask[0,0,0,:5].all(),'源句真 token 位置应该是 True'
    assert not src_mask[0,0,0,5:].any(),'源句 PAD 位置应该是 False'
    assert tgt_mask[0,0,:,4:].sum()==0,'目标句 PAD 位置没被挡住'

    # ---- 3. 下三角:位置 i 看不到它右边的任何位置 ----
    for i in range(len_tgt):
        assert not tgt_mask[0,0,i,i+1:].any(),'第 {} 个位置能看到未来'.format(i)

    # ---- 4. forward 的形状与数值健康 ----
    with torch.no_grad():
        out=model(src,tgt)
    assert out.shape==(batch_size,len_tgt,y_vocab),out.shape
    assert not torch.isnan(out).any(),'输出出现 NaN'

    print('src_mask :',tuple(src_mask.shape))
    print('tgt_mask :',tuple(tgt_mask.shape))
    print('输出 out :',tuple(out.shape))

    # ---- 5. 因果性端到端 ----
    # 只改最后一个真 token(len_tgt=6,后 2 位是 PAD,下标是 3),
    # 它左边位置的输出应当纹丝不动,它自己应当变
    last_real=3
    tgt2=tgt.clone()
    # 在 4..19 里循环挪一位,保证换成的是另一个真 token,不会撞上 PAD
    tgt2[:,last_real]=4+(tgt2[:,last_real]-4+1)%(y_vocab-4)

    with torch.no_grad():
        out2=model(src,tgt2)

    diff_past=(out[:,:last_real]-out2[:,:last_real]).abs().max().item()
    diff_self=(out[:,last_real]-out2[:,last_real]).abs().max().item()

    print('前序位置输出差异:',diff_past,'(应约为 0)')
    print('被改位置输出差异:',diff_self,'(应大于 0)')

    assert diff_past<1e-6,'因果掩码没生效:位置 t 能看到位置 t+1 的输入'
    assert diff_self>0,'改动的输入没传导到输出'
    print('transformer_model 自检通过')