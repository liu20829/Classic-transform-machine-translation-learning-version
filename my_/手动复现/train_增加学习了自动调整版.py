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
import torch
import sys
import my_dictionary
# model/ 里的模块之间用的是平级 import(transformer_model.py 里写的是
# from transformer_decoder import ...),这些名字不在搜索路径上 ——
# 它们自己是入口时才找得到,从上层跑 my_train.py 就不行。
# 在入口处把 model/ 加进来一次即可,不用给每个模块都打补丁。
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'model'))

from model.transformer_model import transformer_model
from my_dataloader import get_dataloader
from my_dictionary import load_or_create

def train_one_epoch(model,train_dataloader,epoch_num,Adam,loss_function,clip,pr_ci):
    total_batchs=len(train_dataloader)
    # pr_ci：一个 epoch 想打多少条中间信息，由调用方给
    # report_every：换算成"每隔多少个 batch 打一条"。
    # max(...,1) 兜底 pr_ci 比 batch 数还大时算出 0（0 会 ZeroDivision 或每条都打）
    # 用整除不用 /：batch 数一旦除不尽，batch_idx % 19.99 永远不等于 0，
    # 会变成只在第 0 个 batch 打一条就再不吭声
    report_every=max(total_batchs//pr_ci,1)
    print("="*78)
    print(f"[train] epoch {epoch_num+1} | batch 数 {total_batchs} | 每 {report_every} 个 batch 报一条   [每行: 本批 / 本 epoch 累计]")
    model.train()
    epoch_loss=0.0      # 累加每个 batch 的平均 loss
    epoch_correct=0     # 累加命中的非 PAD 位置数
    epoch_tokens=0      # 累加参与计数的非 PAD 位置数
    # with tqdm(train_dataloader,desc=f"epoch:[{epoch_num+1}]",unit='batchs') as dataloader:
    for batch_idx,batch in enumerate(train_dataloader):
        # batch=[{sec:[],tgt:[]},{src:[],tgt:[]}]长度等于batch_size
        # 我们要构造的是
        # input=(batch_size,src)
        # target=(batch_size,target)
        input=[]
        target=[]
        batch_size=len(batch)
        # print(batch_size)
        for example in batch:
            # print(example)#{sec:[],tgt:[]}
            input.append(torch.unsqueeze(torch.tensor(example.get("src")).type(torch.int64),dim=0))


            target.append(torch.unsqueeze(torch.tensor(example.get('tgt')).type(torch.int64),dim=0))
            # break
            pass
        input=torch.cat(input,dim=0).to(device)#(batch_size,seq_len)
        # print(input)

        target=torch.cat(target,dim=0).to(device)#(batch_size,seq_len)

        y_p=model(input,target[:,:-1])#target[:,:-1]删掉最后以为构成因果预测掩码
        # print(y_p.shape)#(batch_size,seq_len,tgt_vocab)

        y_p=y_p.contiguous().view(-1,y_p.shape[-1])#(batch_size*seq_len,tgt_vocab)


        target=target[:,1:].contiguous().view(-1)#(batch_size*seq_len,)给展成一维
        Adam.zero_grad()  # 清掉上一轮的梯度
        loss=loss_function(y_p,target)
        loss.backward()  # 算这一轮的梯度
        nn.utils.clip_grad_norm_(model.parameters(),clip)
        Adam.step()  # 用梯度更新参数

        # ---- 统计 ----
        # CrossEntropyLoss 默认 reduction='mean'，且已经按 ignore_index 剔掉 PAD 位置，
        # loss.item() 本身就是"非 PAD 位置的平均"，再除 batch_size 等于平均了两遍
        epoch_loss+=loss.item()

        # 准确率跟 loss 用同一套掩码：只数非 PAD 位置，分母也用位置数，结果才是百分数。
        # 这里的 target 上面已经被展平成 (batch_size*seq_len,)，y_p 也是，两边一一对应
        mask=target!=loss_function.ignore_index#获取有效token索引的掩码
        batch_correct=((y_p.argmax(-1)==target)&mask).sum().item()
        batch_tokens=mask.sum().item()
        epoch_correct+=batch_correct
        epoch_tokens+=batch_tokens

        if batch_idx%report_every==0:
            print(f"[train] {batch_idx:>6}/{total_batchs}  "
                  f"loss {loss.item():7.4f}  acc {100*batch_correct/batch_tokens:6.2f}%  "
                  f"|  {epoch_loss/(batch_idx+1):7.4f}  {100*epoch_correct/epoch_tokens:6.2f}%")
        # break
    epoch_loss/=total_batchs
    print(f"[train] epoch {epoch_num+1} 结束 | 平均 loss {epoch_loss:.4f} | 准确率 {100*epoch_correct/epoch_tokens:.2f}%")
    return epoch_loss
    pass


@torch.no_grad()
def valid_one_epoch(model,valid_dataloader,epoch_num,loss_function,device,pr_ci):
    model.eval()
    epoch_loss,epoch_correct=0.0,0.0
    epoch_tokens=0
    total_batchs = len(valid_dataloader)
    report_every = max(total_batchs // pr_ci, 1)
    print("="*78)
    print(f"[valid] epoch {epoch_num+1} | batch 数 {total_batchs} | 每 {report_every} 个 batch 报一条   [每行: 本批 / 本 epoch 累计]")
    for batch_i,batch in enumerate(valid_dataloader):
        input=[]
        target=[]
        for example in batch:
            input.append(torch.unsqueeze(torch.tensor(example.get('src')).type(torch.int64),dim=0))
            target.append(torch.unsqueeze(torch.tensor(example.get('tgt')).type(torch.int64),dim=0))

            pass
        input=torch.cat(input,dim=0).to(device)
        target=torch.cat(target,dim=0).to(device)

        y_p=model(input,target[:,:-1])

        y_p=y_p.contiguous().view(-1,y_p.shape[-1])

        target=target[:,1:].contiguous().view(-1)

        loss=loss_function(y_p,target)

        epoch_loss+=loss.item()

        mask=target!=loss_function.ignore_index
        batch_correct=((y_p.argmax(-1)==target)&mask).sum().item()
        batch_tokens=mask.sum().item()
        epoch_correct+=batch_correct
        epoch_tokens+=batch_tokens
        if batch_i%report_every==0:
            print(f"[valid] {batch_i:>6}/{total_batchs}  "
                  f"loss {loss.item():7.4f}  acc {100*batch_correct/batch_tokens:6.2f}%  "
                  f"|  {epoch_loss/(batch_i+1):7.4f}  {100*epoch_correct/epoch_tokens:6.2f}%")
    print(f"[valid] epoch {epoch_num+1} 结束 | 平均 loss {epoch_loss/total_batchs:.4f} | 准确率 {100*epoch_correct/epoch_tokens:.2f}%")


def train(model,epoch,batch_size,pad_index,RL,clip,pr_ci):
    # RL = 0.01
    Adam = optim.Adam(model.parameters(), lr=RL)
    # ignore_index忽略填充pad索引
    loss_function=nn.CrossEntropyLoss(ignore_index=pad_index)
    train_dataloader,valid_dataloader,test_dataset=get_dataloader(src_tokens,tgt_tokens,batch_size)
    for epoch_i in range(epoch):
        train_one_epoch(model,train_dataloader,epoch_i,Adam,loss_function,clip,pr_ci)
        valid_one_epoch(model,valid_dataloader,epoch_i,loss_function,device,pr_ci)
        # break
        pass
    pass

if __name__=="__main__":
    import torch
    from my_tokenizer import tokenize
    RL=0.01
    clip=1.0
    pr_ci=5#一个epoch打印多少中间信息
    # ---- 配置：自己复现就自己写全,不依赖 utils ----
    # 路径写法跟 my_dictionary.py 的 __main__ 保持一致:写死绝对路径,看得见读的是哪份文件
    data_path=r"G:\PythonProject\follow-github-learn-agent\Transformer-for-Machine-Translation-main\my_data\WMT_dataset\wmt_zh_en_training_corpus.csv"

    max_pairs=50000#最多读多少行;传 None 会读完整份 6.3GB
    max_sent_len=50#句子最多多少 token;也决定 tokenize 后的序列长度(=它+2)
    min_count=2#出现次数低于此值的词不单独占编号,归入 UNK
    batch_size=20

    model_dim=256#每个token的向量维度
    num_heads=8#注意力头数
    ff_size=1024#解码器ffn中间维度
    feedforward=1024#编码器ffn中间维度
    decoder_num=7#解码器层数
    encoder_num=7#编码器层数
    len_q=50#位置编码的默认长度
    dropout=0.15#随机杀死神经元概率
    padding_index=0#pad的编号
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epoch=2

    # ---- 1. 词表：有缓存直接加载,没有就现建并落盘 ----
    # 缓存文件名里带着 (max_pairs, max_sent_len, min_count),
    # 这三个参数一变文件名就变,会自动重建,不会静默用旧词表
    input_dic,output_dic,src_sents,tgt_sents=load_or_create(
        data_path,max_pairs,max_sent_len,min_count)

    # 必须用 n_count 而不是 len(word2index) —— 后者只数普通词,少 4 个特殊符号,
    # Embedding 行数不够,训练撞到最大编号时才 IndexError
    src_vocab_size=input_dic.n_count     #中文词表大小
    y_vocab_size=output_dic.n_count      #英文词表大小

    # ---- 2. 句子 -> 编号序列 ----
    # 这两个是模块级变量,train() 里能直接读到
    src_tokens=[tokenize(s,input_dic,max_sent_len) for s in src_sents]
    tgt_tokens=[tokenize(s,output_dic,max_sent_len) for s in tgt_sents]

    # 用关键字传参:这个构造函数的形参顺序是打乱的
    # (ff_size 和 feedforward 隔着三个参数,encode_num 甩在最后),位置传参迟早错位
    model=transformer_model(
        src_vocab_size=src_vocab_size,
        model_dim=model_dim,
        num_heads=num_heads,
        y_vocab_size=y_vocab_size,
        ff_size=ff_size,
        decoder_num=decoder_num,
        len_q=len_q,
        feedforward=feedforward,
        dropout=dropout,
        padding_index=padding_index,
        device=device,
        encode_num=encoder_num).to(device)

    train(model,epoch,batch_size,my_dictionary.PAD_TOKEN,RL,clip,pr_ci)
    pass