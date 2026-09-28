"""
用法：
    冒烟测试  python train_ch_en.py --max_pairs 2000 --epoch 2
    正式训练  python train_ch_en.py
"""
import datetime

from torch import nn
import time,os
from torch import optim
from tqdm import tqdm
from torch.nn import functional
import torch
import sys
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import my_dictionary
# model/ 里的模块之间用的是平级 import(transformer_model.py 里写的是
# from transformer_decoder import ...),这些名字不在搜索路径上 ——
# 它们自己是入口时才找得到,从上层跑 my_train.py 就不行。
# 在入口处把 model/ 加进来一次即可,不用给每个模块都打补丁。
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'model'))

from model.transformer_model import transformer_model
from my_dataloader import get_dataloader
from my_dictionary import load_or_create

# ---- TensorBoard（只写标量）----
# 不要对全部参数写 add_histogram：本模型 120M 参数，每轮要落约 481MB 浮点数据，
# 会拖慢训练且对定位问题没帮助（毕设那个 ResNet18 只有 11M，不是一个量级）
WRITER=None      # 由 main() 建好；保持 None 时 _log 静默跳过，方便单独跑某个函数做测试

def _log(tag,value,step):
    """写一条 TensorBoard 标量；WRITER 未初始化时什么都不做"""
    if WRITER is not None:
        WRITER.add_scalar(tag,value,step)

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
        # clip_grad_norm_ 的返回值就是梯度总范数，它本来就算过一遍，接住等于白得一个监控指标
        # （Adam+weight_decay 那次崩盘，这条曲线是第一个会报警的）
        total_norm=nn.utils.clip_grad_norm_(model.parameters(),clip)
        Adam.step()  # 用梯度更新参数
        # 横轴用全局步数，跨 epoch 连成一条线，不会每轮从头开始
        global_step=epoch_num*total_batchs+batch_idx
        _log('Grad/norm',total_norm,global_step)

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
            # 权重范数：嵌入层(77M)和输出层(30.5M)是这个模型最大的两块，崩盘时先变形
            _log('Train/batch_loss',loss.item(),global_step)
            _log('Weights/embed',model.encoder.embedding_positional.embedding.weight.norm().item(),global_step)
            _log('Weights/output',model.decoder.linear.weight.norm().item(),global_step)
        # break
    epoch_loss/=total_batchs
    train_acc=epoch_correct/epoch_tokens
    return epoch_loss,train_acc
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
    valid_loss=epoch_loss/total_batchs
    valid_acc=epoch_correct/epoch_tokens
    return valid_loss,valid_acc

def train(model,epoch,batch_size,pad_index,RL,clip,pr_ci,LR_FACTOR,LR_PATIENCE):
    # RL = 0.01
    # weight_decay=0.01：抑制过拟合（上次 train acc 43.7% vs valid 33.7%），不额外占显存
    # ⚠️ 必须用 AdamW，不能用 Adam(weight_decay=...)。两者不等价：
    #   Adam 把 wd 当 L2 加进梯度，再被 1/(sqrt(v)+eps) 缩放。任务梯度随拟合变小 → sqrt(v)→0
    #   → eps=1e-8 接管分母 → 更新量不降反涨，权重被加速砸向 0。AdamW 是解耦的，不经过这个分母。
    #   [实测 2026-09-28] 一个已拟合(任务梯度≈0)的 Linear(256→1000) 跑 1000 步：
    #     Adam (wd=0)     |W| 18.26 → 17.62
    #     Adam (wd=0.01)  |W| 18.26 →  0.85    每步 |ΔW|=1.74e-2   ← 掉了 95%
    #     AdamW(wd=0.01)  |W| 18.26 → 17.60    每步 |ΔW|=6.57e-4   ← 差 26 倍
    #   对应的真实现象（第一次训练.txt）：epoch1 正常 → epoch2-3 训练 loss 反涨(7.74→9.01) → 再没回到基线
    Adam = optim.AdamW(model.parameters(), lr=RL, weight_decay=0.01)
    scheduler=lr_scheduler.ReduceLROnPlateau(Adam,mode='max',factor=LR_FACTOR,patience=LR_PATIENCE)
    # ignore_index忽略填充pad索引
    # label_smoothing=0.1：当正则用，抑制过度自信。
    # ⚠️ 它会让 loss 数值整体抬高且不再趋近 0，别拿新旧 loss 绝对值对比（acc 口径不变）
    loss_function=nn.CrossEntropyLoss(ignore_index=pad_index,label_smoothing=0.1)
    # seed 必须显式传：seed=None 时两次 train_test_split 各用全局随机流，src/tgt 的排列
    # 各排各的，句对会静默错位——模型只能学到目标语言的词频分布（loss 卡 7.0 / acc 卡 5.48%）
    train_dataloader,valid_dataloader,test_dataset=get_dataloader(src_tokens,tgt_tokens,batch_size,seed=1111)
    best_val_acc=0
    total_train_time=0.0      # 累计训练用时（秒），用来算平均每轮和剩余预估
    # 早停状态：盯 valid loss，跟上面盯 acc 存 best 是两条独立的线
    best_val_loss=float('inf')
    best_loss_epoch=0
    bad_epochs=0
    EARLY_STOP_PATIENCE=8     # 给拐点留余量；发现切早了就调大，代价只是多看几轮

    for epoch_i in range(epoch):
        start_t=time.time()   # 本轮计时起点放在训练开始前：本轮"训练+验证"的耗时都算进去
        # 每个 epoch 开始前打一次当前学习率：scheduler 改的就是 Adam.param_groups 里的 lr，
        # 直接读它才是优化器本轮实际要用的值，降没降一眼可见
        print(f"[train] epoch {epoch_i+1} 开始 | 当前学习率 lr={Adam.param_groups[0]['lr']:.6g}")
        train_loss,train_acc=train_one_epoch(model,train_dataloader,epoch_i,Adam,loss_function,clip,pr_ci)
        valid_loss,valid_acc=valid_one_epoch(model,valid_dataloader,epoch_i,loss_function,device,pr_ci)
        lr_before=Adam.param_groups[0]['lr']      # 记下步进前的值，用来判断这一轮 lr 有没有被改
        # scheduler 只吃准确率：mode='max' 盯的就是它，误喂 valid_loss 会被判成每轮都没提升，lr 会莫名连降
        scheduler.step(valid_acc)
        lr_after=Adam.param_groups[0]['lr']
        # ReduceLROnPlateau 的实际触发条件是连续 patience+1 个 epoch 没提升（内部判断 num_bad_epochs > patience）
        if lr_after!=lr_before:
            print(f"[lr] 学习率调整：{lr_before:.6g} → {lr_after:.6g}   [验证准确率连续 {LR_PATIENCE+1} 个 epoch 未提升]")
        print(f"[epoch {epoch_i+1}] 汇总 | train loss {train_loss:7.4f} acc {100*train_acc:6.2f}%"
              f" | valid loss {valid_loss:7.4f} acc {100*valid_acc:6.2f}%")
        # 轮级标量：以 epoch 为横轴。train/valid 两条放同一张图才看得出过拟合的张口
        _log('Loss/train_epoch',train_loss,epoch_i+1)
        _log('Loss/valid_epoch',valid_loss,epoch_i+1)
        _log('Acc/train_epoch',100*train_acc,epoch_i+1)
        _log('Acc/valid_epoch',100*valid_acc,epoch_i+1)
        _log('LR',Adam.param_groups[0]['lr'],epoch_i+1)
        # ---- 计时与预估：本轮用时、累计用时、按平均轮时长推剩余 ----
        e_time=time.time()-start_t              # 本轮用时（训练+验证）
        total_train_time+=e_time
        avg_time=total_train_time/(epoch_i+1)   # 已跑完轮次的平均用时，比只拿第一轮估更稳
        eta=avg_time*(epoch-epoch_i-1)          # 预估剩余 = 平均每轮 × 剩余轮数
        print(f"[time] 本轮用时 {e_time:.1f}s | 累计用时 {total_train_time/60:.2f} min"
              f" | 预计剩余 {eta/60:.2f} min | 预计总时长 {avg_time*epoch/60:.2f} min")
        # break
        if valid_acc>best_val_acc:
            best_val_acc=valid_acc
            # 编号用 epoch_i+1（当前轮）：用形参 epoch 是总轮数 10，文件名和 ckpt 里会永远写成 10
            best_name=f"best_epoch{epoch_i+1:02d}_acc{valid_acc * 100:.2f}.pth"
            ckpt={
                "epoch":epoch_i+1,
                "model_state":model.state_dict(),
                "optimizer_state":Adam.state_dict(),
                "val_acc":valid_acc
            }

            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, best_name))
            torch.save(ckpt, BEST_MODEL_PATH)
            print(f"🌟 [Save] 发现更优模型: {best_name}")
        # ---- 早停：valid loss 连续 N 轮不创新低就收工 ----
        # 判据用 loss 不用 acc：acc 抖，且过拟合后还会缓慢爬（run.log 里 50 轮从 32.8% 爬到 33.8%），
        # 拿它当判据几乎不触发——那个 lr 调度器就是这么被拖着连砍 6 次、后期空转的
        # 放在存 best 之后：停之前这一轮如果 acc 更好，先存完再退出
        if valid_loss<best_val_loss:
            best_val_loss=valid_loss
            best_loss_epoch=epoch_i+1
            bad_epochs=0
        else:
            bad_epochs+=1
            if bad_epochs>=EARLY_STOP_PATIENCE:
                print(f"[stop] valid loss 连续 {bad_epochs} 轮未创新低"
                      f"（最低 {best_val_loss:.4f} @ epoch {best_loss_epoch}），提前结束")
                break
    # 收尾：不 close 的话最后一段数据可能还留在写缓冲区里没落盘
    if WRITER is not None:
        WRITER.close()
def main():
    import torch
    from my_tokenizer import tokenize
    global log_root, CHECKPOINT_DIR, runs_root, BEST_MODEL_PATH, LAST_MODEL_PATH, LOG_CSV,src_tokens,tgt_tokens,device,WRITER
    RL=0.0001
    clip=1.0
    pr_ci=5#一个epoch打印多少中间信息
    # ✅ 在主函数中设置动态日志目录
    log_root = "base_trian_log_" + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    os.makedirs(log_root, exist_ok=True)
    # ✅ 根据 log_root 设置其他路径
    CHECKPOINT_DIR = os.path.join(log_root, "checkpoints")
    # torch.save 不会自动创建父目录：只拼路径不建目录，第一次保存 best 就会 RuntimeError
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # TensorBoard 事件目录跟 checkpoints 并排，都在这一轮的 log_root 下
    WRITER = SummaryWriter(log_dir=os.path.join(log_root, "tb"))
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet18_9.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_resnet18_9.pth")
    # 调度器参数
    LR_FACTOR = 0.5
    # 5 → 12：上次 patience=5 时 lr 在 epoch 41 前后就被砍到 6.25e-6 冻结，后面 60 个 epoch 基本空转
    LR_PATIENCE = 12
    # ---- 配置：自己复现就自己写全,不依赖 utils ----
    # 路径写法跟 my_dictionary.py 的 __main__ 保持一致:写死绝对路径,看得见读的是哪份文件
    data_path=r"/home/AI_agent/lll2/my_/train_translation/dataset/wmt_zh_en_training_corpus.csv"

    max_pairs=1000000#最多读多少行;传 None 会读完整份 5.9GB（全语料 24752393 行）
    # [实测 2026-09-24] 超长丢弃率极低，不是此前注释写的"四成"：
    #   20万→199932对(丢68)  100万→999548对  200万→1970365对    丢弃率 0.03%
    #   100万对实测：zh词表 181564 / en词表 119110，token 后内存 1.37GB，准备 50s
    # 数据量是压过拟合最有效的一招：run.log 那轮只用了 20 万对(占全语料 0.8%)就过拟合
    max_sent_len=50#句子最多多少 token;也决定 tokenize 后的序列长度(=它+2)
    min_count=2#出现次数低于此值的词不单独占编号,归入 UNK
    batch_size=128#20 → 128：batch 20 时每 batch 耗时(约0.09s)和 128 一样，
                  #说明瓶颈在 CPU 侧的 tensor 拼装、GPU 闲着，吞吐差 6.4 倍
                  #（20→约200样本/s，128→约1230样本/s）。显存按 en 词表 119110 估约 8.9G

    # 512/16/2048/14+14 → 256/8/1024/7+7（撤掉翻倍）：
    # ① 与上次 33.7% 的基线同构，数据量实验才有可比性；② 翻倍版估 ~3.6~4GB，4GB 卡上大概率 OOM；
    # ③ 过拟合已确认（train 43.7% vs valid 33.7%），扩容只会放大它。要扩也是"先扩数据"，不是先扩模型
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
    epoch=30#上限，不是计划：valid loss 见拐点后 8 轮会被早停接走（run.log 拐点在 epoch 29）
            # 每轮 = 80万样本/128 ≈ 6250 train batch + 1250 valid batch
            # 按 run.log 实测的 0.087 s/batch 直推约 11 min；但 en 词表从 57942 涨到 119110，
            # 输出层变重，实际会更慢——这是估计，第一轮跑完就有准数

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

    train(model,epoch,batch_size,my_dictionary.PAD_TOKEN,RL,clip,pr_ci,LR_FACTOR,LR_PATIENCE)
    pass
if __name__=="__main__":
    main()