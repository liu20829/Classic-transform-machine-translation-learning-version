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

# model/ 里的模块之间用的是平级 import(transformer_model.py 里写的是
# from transformer_decoder import ...),这些名字不在搜索路径上 ——
# 它们自己是入口时才找得到,从上层跑 my_train.py 就不行。
# 在入口处把 model/ 加进来一次即可,不用给每个模块都打补丁。
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'model'))

from transformer_model import transformer_model
from my_dataloader import get_dataloader
from my_dictionary import load_or_create

def train_one_epoch(model,train_dataloader,batch_size,Adam):
    print("-"*50)
    print(f"[train]训练开始| batch数：{len(train_dataloader)}")
    model.train()
    runing_loss,runing_acc,total=0.0,0.0,0
    for batch_idx,batch in enumerate(train_dataloader):
        # batch=[{sec:[],tgt:[]},{src:[],tgt:[]}]长度等于batch_size
        # 我们要构造的是
        # input=(batch_size,src)
        # target=(batch_size,target)
        input=[]
        target=[]
        for example in batch:
            input.append(example.get("src"))
            target.append(example.get('tgt'))
            pass

        break
    pass
def valid_one_epoch(model,valid_dataloader):

    pass
def train(model,epoch,batch_size):
    RL = 0.01
    Adam = optim.Adam(model.parameters(), lr=RL)
    train_dataloader,valid_dataloader,test_dataset=get_dataloader(src_tokens,tgt_tokens,batch_size)
    for epoch_i in range(epoch):
        train_one_epoch(model,train_dataloader,batch_size,Adam)
        valid_one_epoch(model,valid_dataloader)
        break
        pass
    pass

if __name__=="__main__":
    import torch
    from my_tokenizer import tokenize

    # ---- 配置：自己复现就自己写全,不依赖 utils ----
    # 路径写法跟 my_dictionary.py 的 __main__ 保持一致:写死绝对路径,看得见读的是哪份文件
    data_path=r"G:\PythonProject\follow-github-learn-agent\Transformer-for-Machine-Translation-main\my_data\WMT_dataset\wmt_zh_en_training_corpus.csv"

    max_pairs=200000#最多读多少行;传 None 会读完整份 6.3GB
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

    train(model,epoch,batch_size)
    pass