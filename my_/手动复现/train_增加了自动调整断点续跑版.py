"""
用法：
    冒烟测试        python train_增加了自动调整断点续跑版.py --max_pairs 2000 --epochs 2
    正式训练        python train_增加了自动调整断点续跑版.py
    断点续跑        python train_增加了自动调整断点续跑版.py --resume
    续跑指定那一轮  python train_增加了自动调整断点续跑版.py --resume base_trian_log_09-28_10-00-00
    接着多练几轮    python train_增加了自动调整断点续跑版.py --resume --epochs 40
    从头重跑        python train_增加了自动调整断点续跑版.py --fresh

断点续跑说明：
    每个 epoch 跑完都会把 last.pth 覆盖写一次，里面装的是"下一轮从哪继续"所需的
    全部状态：模型、AdamW、lr 调度器、随机数发生器、早停计数、best 指标。
    所以 --resume 之后接着跑，等价于那次训练一直没停过（数据切分固定 seed，
    shuffle 的随机流也从 last.pth 里恢复，第 N+1 轮的 batch 顺序与原来一致）。
    ⚠️ 断点里同时存了 (batch_size / 词表大小 / src_tgt 条数)，续跑时对不上会警告：
       改了数据或词表就等于换了一道题，参数接着用没有意义，该 --fresh 重跑。
"""
import datetime

from torch import nn
import time,os
from torch import optim
from tqdm import tqdm
from torch.nn import functional
import torch
import sys
import argparse
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
SCALER=None      # 混合精度的 GradScaler，train() 里建好；没开 amp 时保持 None

def _log(tag,value,step):
    """写一条 TensorBoard 标量；WRITER 未初始化时什么都不做"""
    if WRITER is not None:
        WRITER.add_scalar(tag,value,step)

# ---- 断点续跑：命令行参数 ----
def parse_args(argv=None):
    parser=argparse.ArgumentParser(
        description="中英翻译 Transformer 训练（支持从断点继续）")
    # nargs='?': --resume 后面可以不跟路径
    #   python x.py --resume                                -> 'auto'，自动找最近一次
    #   python x.py --resume base_trian_log_09-28_10-00-00  -> 就用这一轮（目录或 last.pth 都行）
    #   python x.py                                         -> None，全新一轮（日志目录带时间戳，不会覆盖别人）
    parser.add_argument('--resume',nargs='?',const='auto',default=None,metavar='RUN_DIR',
                        help="从断点继续训练；不带值时自动挑最近一次训练留下的断点")
    parser.add_argument('--log_dir',default=None,metavar='DIR',
                        help="配合 --resume 指定续到哪一轮；只给 --log_dir 且目录里有断点时会直接续跑")
    parser.add_argument('--fresh',action='store_true',
                        help="即使目录里有断点也从第 1 轮重新开始")
    parser.add_argument('--epochs',type=int,default=None,
                        help="总轮数上限；默认沿用脚本里的 30，可用来接着多跑几轮")
    parser.add_argument('--max_pairs',type=int,default=None,
                        help="最多读多少行语料；默认 10000000（约 1000 万对）")
    parser.add_argument('--batch_size',type=int,default=None,
                        help="批大小；默认 512。显存不够就按 256/128 往下退")
    parser.add_argument('--lr',type=float,default=None,
                        help="学习率；默认 2e-4。跟 batch_size 成比例改（线性缩放）")
    parser.add_argument('--min_count',type=int,default=None,
                        help="词频低于此值的词归入 UNK；默认 3")
    parser.add_argument('--num_workers',type=int,default=None,
                        help="DataLoader 子进程数；默认 8。设为 0 就是主进程串行（原来的行为）")
    parser.add_argument('--data_path',default=None,
                        help="语料 CSV 路径；默认脚本里写死的那份")
    # argparser 里 True/False 的成对开关用同一个 dest，谁在后面谁生效
    parser.add_argument('--no-amp',dest='amp',action='store_false',
                        help="关掉混合精度（排查数值问题、或在不支持 bf16 的老卡上跑时用）")
    parser.add_argument('--amp',dest='amp',action='store_true',
                        help="开混合精度（默认开）")
    parser.set_defaults(amp=True)
    return parser.parse_args(argv)

def _dur(sec):
    """秒数按量级换单位：一轮只跑 0.2s 时写 '0.0s' 看不出快慢，冒烟测试那种小数据尤其明显"""
    if sec<1:
        return f"{sec*1000:.0f}ms"
    if sec<60:
        return f"{sec:.1f}s"
    if sec<3600:
        return f"{sec/60:.2f}min"
    return f"{sec/3600:.2f}h"

def _pct(x):
    """断点里可能压根没记 best（旧版断点）或记成 ±inf：直接格式化会显示成刺眼的 0.00%"""
    try:
        x=float(x)
    except (TypeError,ValueError):
        return "未知"
    return "未知" if x!=x or x in (float('inf'),float('-inf')) else f"{x*100:.2f}%"

def _num(x,spec=".4f"):
    """同上，给 loss 用：inf 显示成"未知"而不是 inf"""
    try:
        x=float(x)
    except (TypeError,ValueError):
        return "未知"
    return "未知" if x!=x or x in (float('inf'),float('-inf')) else format(x,spec)

def _load_ckpt(path):
    """读断点。优先 weights_only=True（只反序列化张量，不会被 .pth 里的代码执行到）"""
    try:
        return torch.load(path,map_location='cpu',weights_only=True)
    except Exception:
        # 断点里有 numpy/CUDA 随机状态这类非张量对象，weights_only=True 读不了，退回完整反序列化。
        # 只读自己训练时存的 last.pth 没问题；别人给的 .pth 不要这么读
        return torch.load(path,map_location='cpu',weights_only=False)

def _elapsed_from(ckpt):
    """断点里累计的训练用时（秒）。老断点没这个字段就返回 0.0，别让续跑直接崩"""
    meta=ckpt.get('meta') or {}
    return float(meta.get('total_train_time',0.0) or 0.0)

def _scan_log_root(root,cands):
    """扫一个目录下的 base_trian_log_*，把 (断点落盘时间, 断点路径) 追加进 cands"""
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        if not name.startswith('base_trian_log_'):
            continue
        # 断点可能直接躺在 base_trian_log_*/ 下，也可能在它下面的 checkpoints/ 里
        newest=None
        for cdir in (os.path.join(root,name,'checkpoints'),os.path.join(root,name)):
            if os.path.isdir(cdir):
                newest=_pick_in_dir(cdir)
            if newest is not None:
                break
        if newest is not None:
            cands.append((os.path.getmtime(newest),newest))

def find_latest_run(root=None):
    """
    找最近一次训练的断点：扫 base_trian_log_*，按断点的落盘时间挑最新的一个。

    ⚠️ 要同时扫"脚本所在目录"和"当前工作目录"，不能只看一个：
       日志目录名是相对路径（log_root="base_trian_log_..."），落在 **cwd** 下；
       但如果平时习惯 cd 到脚本目录再跑，两者就是同一个。只认一个的话，
       从别处 `python my_/手动复现/xxx.py --resume` 会出现"明明有断点却说没找到"
       —— 实测就踩到过：日志在仓库根，脚本在 my_/手动复现/，auto 一路报没找到。
    root 显式传时就只扫它（自检用，行为可预期）。
    """
    roots=[root] if root is not None else [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.getcwd()),
    ]
    cands=[]
    for r in roots:
        _scan_log_root(r,cands)
    if not cands:
        return None
    # 同一份断点可能被两个 root 扫到，按路径去重后再比时间。
    # 并列也稳：mtime 相同时按路径排序取最后一个，不会这次挑 A 下次挑 B
    uniq={}
    for mtime,path in cands:
        uniq[path]=mtime
    return sorted((m,p) for p,m in uniq.items())[-1][1]

def _pick_in_dir(cdir):
    """
    在一个 checkpoints 目录里挑断点：优先 last.pth（每轮覆盖写），没有就退回到别的。
    老脚本 train_增加学习了自动调整版.py 只会写 best_resnet18_9.pth 和
    best_epochNN_accXX.pth（从没写过 last.pth），所以这几种都要认得，
    否则"从老训练续跑"就得手工敲完整文件名
    """
    last=os.path.join(cdir,'last.pth')
    if os.path.isfile(last):
        return last
    # 固定名的最优模型（老脚本每创新高覆写一次，存的是历史最优那轮的权重）
    for fixed in ('best_resnet18_9.pth','best.pth'):
        p=os.path.join(cdir,fixed)
        if os.path.isfile(p):
            return p
    # 最后退回到 best_epochNN_accXX.pth 里落盘最晚的那个
    snaps=[os.path.join(cdir,f) for f in os.listdir(cdir)
           if f.startswith('best_epoch') and f.endswith('.pth')]
    return max(snaps,key=os.path.getmtime) if snaps else None

def resolve_resume(args):
    """把命令行意图翻译成"从哪个断点续"，顺带定下这一轮的日志目录"""
    spec=args.resume
    if spec is None and args.log_dir is not None and not args.fresh:
        # 给了目录就按目录里的断点续，省得再写一遍 --resume
        spec=args.log_dir
    if spec is None or args.fresh:
        return None,None

    if spec=='auto':
        ckpt=find_latest_run()
        if ckpt is None:
            print("[resume] 没找到任何带断点的 base_trian_log_*，这次按全新训练开始")
            return None,None
    else:
        if os.path.isdir(spec):
            # 给目录时：先看 spec/checkpoints/，再退一步直接看 spec/ 自己
            ckpt=_pick_in_dir(os.path.join(spec,'checkpoints')) or _pick_in_dir(spec)
            if ckpt is None:
                print(f"[resume] {spec} 里没找到断点文件，这次按全新训练开始")
                return None,None
        elif os.path.isfile(spec):
            ckpt=spec
        else:
            # 路径写错了还闷头从头训，是最气人的一种失败：直接停
            raise SystemExit(f"[resume] 找不到断点：{spec}")

    reload_ckpt=_load_ckpt(ckpt)
    if 'model_state' not in reload_ckpt:
        raise SystemExit(f"[resume] {ckpt} 不是训练断点（里面没有 model_state）")
    # 日志目录固定成那一轮自己的目录：TensorBoard 的标量续着写，曲线不会断成两张图
    log_dir=args.log_dir or os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    return ckpt,log_dir

def _restore_scaler(scaler,ckpt):
    """
    恢复混合精度的动态损失缩放状态。
    不恢复也能跑，但缩放系数会从初始值重新爬 —— 续跑头几步可能出现一次
    不必要的"梯度溢出 -> 跳过这一步"，loss 曲线会抖一下。
    torch 版本差异：新的 torch.amp.GradScaler 用 set_state_dict/state_dict()，
    老的是 load_state_dict()，两个都兜住。
    """
    if scaler is None or 'scaler_state' not in ckpt:
        return
    try:
        if hasattr(scaler,'set_state_dict'):
            scaler.set_state_dict(ckpt['scaler_state'])
        else:
            scaler.load_state_dict(ckpt['scaler_state'])
    except Exception as e:
        print(f"[resume] 警告：混合精度的缩放状态没装上（{e}），缩放系数从头开始爬")

def restore_for_resume(path,model,Adam,scheduler,device,scaler=None):
    """
    把断点里的状态装回内存，返回起始轮次。
    起始轮次按 0 计：断点 epoch=5 表示 1~5 轮已跑完，要从第 6 轮（下标 5）接着跑。
    累计用时不在返回值里 —— 调用方自己从断点 meta 里取（见 _elapsed_from），
    这样"恢复状态"和"读元信息"是两件事，不会因为谁多返回一个值就串味。
    """
    ckpt=_load_ckpt(path)
    status=ckpt.get('status','interrupted')
    done_epoch=int(ckpt.get('epoch',0) or 0)

    model.load_state_dict(ckpt['model_state'])
    # AdamW 的动量、平方梯度、以及 param_groups 里的当前 lr 都在这里
    if 'optimizer_state' in ckpt:
        Adam.load_state_dict(ckpt['optimizer_state'])
    else:
        print("[resume] 警告：断点里没有 optimizer_state，优化器动量从零开始（loss 会有一次小回弹）")

    if 'scheduler_state' in ckpt:
        # ReduceLROnPlateau 的 state_dict() 不含 best/num_bad_epochs —— 那两个正是
        # "连续几轮没提升该砍 lr"的计数器。不恢复的话刚续跑就重新计数，
        # 该降的 lr 迟迟不降（patience=12 时白跑十几轮）
        try:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        except Exception as e:
            print(f"[resume] 警告：调度器状态没装上（{e}），lr 保持断点值但耐心计数从零开始")
    else:
        print("[resume] 警告：断点里没有 scheduler_state，lr 调度器的耐心计数从零开始")
    # 恢复后再打一次：load_state_dict 不会覆盖 Adam.param_groups 里已有的 lr，
    # 上面那次优化器加载才是 lr 的真实来源，这里打印是为了让人一眼看见续的是多少
    print(f"[resume] 学习率恢复为 lr={Adam.param_groups[0]['lr']:.6g}")

    # 混合精度的缩放状态（没开 amp 或断点里没有就什么都不做）
    _restore_scaler(scaler,ckpt)

    if 'rng_state' in ckpt:
        rng=ckpt['rng_state']
        try:
            # 这里才是"接着跑"和"重跑"的区别所在：DataLoader(shuffle=True) 没有自己的
            # generator，洗牌用的是全局 RNG。不恢复它，第 N+1 轮的 batch 顺序会重新抽，
            # 断点前后就不是同一条训练轨迹了（loss 曲线会跳一下）
            t=rng['torch']
            torch.set_rng_state(t.cpu() if torch.is_tensor(t) else t)
            if device.type=='cuda' and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
        except Exception as e:
            print(f"[resume] 警告：随机数状态没恢复（{e}），第 {done_epoch+1} 轮的 batch 顺序会重新洗")
    else:
        print("[resume] 警告：断点里没有 rng_state，batch 顺序会重新洗")

    print(f"[resume] 从 {path} 续跑")
    print(f"[resume] 断点记录：已完成 {done_epoch} 轮 | val_acc {_pct(ckpt.get('val_acc',float('nan')))}"
          f" | best_val_acc {_pct(ckpt.get('best_val_acc',float('nan')))}"
          f" | best_val_loss {_num(ckpt.get('best_val_loss'))}"
          f" | 早停计数 {ckpt.get('bad_epochs',0)}/{ckpt.get('early_stop_patience','?')}"
          f" | 断点类型 {status}")
    if status=='finished':
        print("[resume] 提示：这个断点是正常跑完（或早停）后留下的，续跑等于接着再练几轮；"
              "想多跑几轮可以配合 --epochs")
    # 只回轮次。断点里的 best_val_acc 是"历史最好"，调用方直接取过来用，
    # 不会因为续跑一次被重置成 0，再白存一遍本来就更好的模型
    return done_epoch

def check_resume_config(saved,config,batch_size):
    """
    把断点里记的配置和本次要用的对一遍，返回不一致的清单。
    单独拎成函数是为了在"没有可跑的轮次"时也能跑一遍：那种情况以前直接跳过检查，
    配置换了却一个字都不报，比从头训还省事地错下去。
    """
    meta=saved.get('meta') or {}
    pairs=[
        ('batch_size',meta.get('batch_size'),batch_size),
        ('src_vocab_size',meta.get('src_vocab_size'),config.get('src_vocab_size') if config else None),
        ('y_vocab_size',meta.get('y_vocab_size'),config.get('y_vocab_size') if config else None),
        ('num_pairs',meta.get('num_pairs'),config.get('num_pairs') if config else None),
    ]
    return [f"{k}(断点 {a} → 本次 {b})" for k,a,b in pairs
            if a is not None and b is not None and a!=b]

def save_checkpoint(path,model,Adam,scheduler,epoch,val_acc,state):
    """
    把"下一轮从哪继续"所需的状态整包写到 path。
    先写临时文件再 replace：中途断电/被 kill 时，last.pth 要么还是上一轮的完整版，
    要么已是新写的完整版，不会留下半截文件让续跑直接报错。
    """
    ckpt={
        'epoch':epoch,                    # 已经跑完的轮数，续跑从 epoch+1 轮开始
        'model_state':model.state_dict(),
        'optimizer_state':Adam.state_dict(),
        'scheduler_state':scheduler.state_dict(),
        'val_acc':val_acc,
        'best_val_acc':state['best_val_acc'],
        'best_val_loss':state['best_val_loss'],
        'best_loss_epoch':state['best_loss_epoch'],
        'bad_epochs':state['bad_epochs'],
        'early_stop_patience':state['early_stop_patience'],
        # 随机数状态：DataLoader 的 shuffle 和 dropout 都吃它，不存就续不成同一条轨迹
        'rng_state':{
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        # 混合精度的动态损失缩放状态（没开 amp 时写 None）。
        # 是纯 Python 的 dict/list/float，不影响 weights_only=True 反序列化
        'scaler_state':(SCALER.state_dict()
                        if SCALER is not None and hasattr(SCALER,'state_dict') else None),
        'status':state['status'],         # interrupted / finished
        # meta 给续跑时核对配置用。里面有 dict 和 str，所以 weights_only=True 读不动，
        # 由 _load_ckpt 自动退回完整反序列化。
        # 配置项统一走 state 取（_save_state 已经把 config 并进去了），省得每个字段都写两遍
        'meta':{
            'batch_size':state.get('batch_size'),
            'src_vocab_size':state.get('src_vocab_size'),
            'y_vocab_size':state.get('y_vocab_size'),
            'num_pairs':state.get('num_pairs'),
            'max_sent_len':state.get('max_sent_len'),
            'min_count':state.get('min_count'),
            'max_pairs':state.get('max_pairs'),
            'epochs_planned':state.get('epochs_planned'),
            'total_train_time':state.get('total_train_time'),
            'saved_at':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    tmp=path+'.tmp'
    torch.save(ckpt,tmp)
    os.replace(tmp,path)

def _to_batch(batch,device):
    """
    把 dataloader 吐出的 [{src:[...],tgt:[...]}, ...] 拼成两个 (batch_size, seq_len) 张量。

    ⚠️ 这里原来是"每个样本 torch.tensor -> unsqueeze -> torch.cat"，一个 batch 要做
    128 次 Python 迭代 + 128 次 torch.tensor 调用。瓶颈其实在 CPU 侧的拼装而不是 GPU:
    脚本里实测过 batch 20 和 batch 128 每步耗时一样（约 0.09s），说明 GPU 一直在等 CPU。
    换成一次 torch.tensor(嵌套列表) 直接出整块：tokenize 保证每句长度恒为 max_sent_len+2，
    本来就不需要 pad/collate，也就没理由一个样本一个样本地单独建张量。

    non_blocking=True 要配合 DataLoader(pin_memory=True) 才有意义（固定内存可以直接
    异步拷到显卡），没开 pin_memory 时它退化成同步拷贝，不会有副作用。
    """
    src=torch.tensor([e['src'] for e in batch],dtype=torch.int64)
    tgt=torch.tensor([e['tgt'] for e in batch],dtype=torch.int64)
    return src.to(device,non_blocking=True),tgt.to(device,non_blocking=True)

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
        input,target=_to_batch(batch,device)

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

def train_one_epoch(model,train_dataloader,epoch_num,Adam,loss_function,clip,pr_ci,
                    scaler=None,amp_dtype=None):
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
    use_amp=amp_dtype is not None
    # with tqdm(train_dataloader,desc=f"epoch:[{epoch_num+1}]",unit='batchs') as dataloader:
    for batch_idx,batch in enumerate(train_dataloader):
        # batch=[{src:[],tgt:[]},...]，长度等于 batch_size
        input,target=_to_batch(batch,device)     # 一次拼成 (batch_size,seq_len)
        # y_p 的形状 (batch_size,seq_len,tgt_vocab)：seq_len=51 时这一个张量就是
        # 128×51×119110×4B ≈ 3GB，是整个训练显存的大头，比模型参数本身重得多
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
            y_p=model(input,target[:,:-1])#target[:,:-1]删掉最后一位构成因果预测掩码
            # print(y_p.shape)#(batch_size,seq_len,tgt_vocab)

            y_p=y_p.contiguous().view(-1,y_p.shape[-1])#(batch_size*seq_len,tgt_vocab)


            target_flat=target[:,1:].contiguous().view(-1)#(batch_size*seq_len,)给展成一维
            loss=loss_function(y_p,target_flat)

        Adam.zero_grad()  # 清掉上一轮的梯度
        if use_amp:
            # GradScaler：fp16 动态损失缩放。梯度太小会下溢成 0，它自动放大 loss 再缩回来。
            # bf16 的动态范围和 fp32 一样，其实不需要缩放，但走同一条路代码更简单
            # （scaler 在 bf16 下会自己判断不缩放）。CPU 上没有 scaler 也能跑，见 main()
            scaler.scale(loss).backward()
            # ⚠️ 必须先 unscale_ 再 clip：clip_grad_norm_ 量的是真实梯度范数，
            #    带着缩放系数去裁，阈值 clip=1.0 就形同虚设
            scaler.unscale_(Adam)
        else:
            loss.backward()
        # clip_grad_norm_ 的返回值就是梯度总范数，它本来就算过一遍，接住等于白得一个监控指标
        # （Adam+weight_decay 那次崩盘，这条曲线是第一个会报警的）
        total_norm=nn.utils.clip_grad_norm_(model.parameters(),clip)
        if use_amp:
            scaler.step(Adam)
            scaler.update()
        else:
            Adam.step()  # 用梯度更新参数
        # 横轴用全局步数，跨 epoch 连成一条线，不会每轮从头开始
        global_step=epoch_num*total_batchs+batch_idx
        _log('Grad/norm',total_norm,global_step)

        # ---- 统计 ----
        # CrossEntropyLoss 默认 reduction='mean'，且已经按 ignore_index 剔掉 PAD 位置，
        # loss.item() 本身就是"非 PAD 位置的平均"，再除 batch_size 等于平均了两遍
        # ⚠️ 统计必须用未缩放的 loss：scaler.scale() 放大了 loss，拿它累加会得到
        #    一个随缩放系数漂移的假曲线
        epoch_loss+=loss.item()

        # 准确率跟 loss 用同一套掩码：只数非 PAD 位置，分母也用位置数，结果才是百分数。
        # target_flat 是 (batch_size*seq_len,)，y_p 也是，两边一一对应
        mask=target_flat!=loss_function.ignore_index#获取有效token索引的掩码
        batch_correct=((y_p.argmax(-1)==target_flat)&mask).sum().item()
        batch_tokens=mask.sum().item()
        epoch_correct+=batch_correct
        epoch_tokens+=batch_tokens

        if batch_idx%report_every==0:
            print(f"[train] {batch_idx:>6}/{total_batchs}  "
                  f"loss {loss.item():7.4f}  acc {100*batch_correct/batch_tokens:6.2f}%  "
                  f"|  {epoch_loss/(batch_idx+1):7.4f}  {100*epoch_correct/epoch_tokens:6.2f}%")
            _log('Train/batch_loss',loss.item(),global_step)
            # 权重范数：崩盘时先变形的是嵌入层那一大块。
            # ⚠️ tying 之后输出层和 decoder 词嵌入是同一份参数，
            #    原来那条 Weights/output 曲线和 Weights/decoder_embed 会完全重合，
            #    所以这里改成盯"源嵌入（中文，最大的一块）"和"解码器第 0 层的 FFN 首层"
            #    （feedforwardlayer 里是 nn.Sequential，线性层在 ff_layer[0]）
            _log('Weights/encoder_embed',model.encoder.embedding_positional.embedding.weight.norm().item(),global_step)
            _log('Weights/decoder_ffn',
                 model.decoder.transformer_decoder_num[0].feed_forward.ff_layer[0].weight.norm().item(),
                 global_step)
        # break
    epoch_loss/=total_batchs
    train_acc=epoch_correct/epoch_tokens
    return epoch_loss,train_acc
    pass


def train(model,epoch,batch_size,pad_index,RL,clip,pr_ci,LR_FACTOR,LR_PATIENCE,
          resume_path=None,config=None,num_workers=4,pin_memory=True,
          amp_dtype=None):
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
    #
    # num_workers：原来没传，默认是 0 —— 也就是"拼 tensor 的活在主进程里串行做"。
    # 脚本里自己记过"batch 20 和 128 每步耗时一样，GPU 闲着"，根子就在这。
    # 换成多进程预取之后主进程只管喂显卡，CPU 侧的拼装被摊到子进程。
    # ⚠️ num_workers>0 必须保证入口有 if __name__=="__main__" 保护（本文件有），
    #    否则子进程会重新 import 一遍本模块、再开一次训练
    train_dataloader,valid_dataloader,test_dataset=get_dataloader(
        src_tokens,tgt_tokens,batch_size,seed=1111,
        num_workers=num_workers,pin_memory=pin_memory)
    best_val_acc=0
    total_train_time=0.0      # 累计训练用时（秒），用来算平均每轮和剩余预估
    # 早停状态：盯 valid loss，跟上面盯 acc 存 best 是两条独立的线
    best_val_loss=float('inf')
    best_loss_epoch=0
    bad_epochs=0
    EARLY_STOP_PATIENCE=8     # 给拐点留余量；发现切早了就调大，代价只是多看几轮
    start_epoch=0             # 从第几轮开始（断点续跑时会被改成断点记的轮数）
    saved={}                  # 断点内容；没续跑时留空

    # ---- 混合精度 ----
    # amp_dtype=None 表示不启用；否则是 torch.float16 / torch.bfloat16。
    # 为什么要它：y_p 那个 (B, L, V) 张量是显存大头，bf16 直接砍一半，
    # 顺带在 4090 这种卡上把矩阵乘加速。
    # scaler 只在 CUDA 上才有意义（CPU 上 torch.amp.GradScaler 会退化成不缩放的透传）。
    # bf16 的动态范围和 fp32 一样，本来不需要损失缩放，enabled=False 即可，
    # 但代码路径保持一致，少一套分支就少一处出错的地方
    use_amp=amp_dtype is not None
    if use_amp:
        scaler=torch.amp.GradScaler(device.type,
                                    enabled=(device.type=='cuda' and amp_dtype==torch.float16))
    else:
        scaler=None
    global SCALER
    SCALER=scaler            # _save_state 靠它把缩放状态写进断点，不用在调用处传一路

    # ---- 断点续跑：把模型/优化器/调度器/随机状态装回来 ----
    # 位置放在这里而不是 main()：此刻 Adam、scheduler、dataloader 都已经建好，
    # 再靠前一点就无对象可恢复
    if resume_path is not None:
        start_epoch=restore_for_resume(resume_path,model,Adam,scheduler,device,scaler)
        saved=_load_ckpt(resume_path)
        total_train_time=_elapsed_from(saved)   # 断点之前累计的训练用时，接着往上加
        # ---- 兼容旧版断点 ----
        # train_增加学习了自动调整版.py 存的是 {epoch, model_state, optimizer_state, val_acc}，
        # 只有权重和动量，best_* / 早停计数都没存。这里要能直接接着跑，不能一读就崩。
        # ⚠️ 关键：这种情况 best_val_acc 不能用 0.0 兜底。旧断点存的是**历史最优那轮**的模型，
        #   它当时的 acc 肯定 >0，用 0.0 当历史最好，续跑后每一轮都会判成"创新高"，
        #   等于每轮白存一份 best 文件、best 的含义也乱了。用 -inf 起手，
        #   让续跑后第一轮先把真实分数重新记下来，之后立刻恢复正常
        legacy='best_val_acc' not in saved
        best_val_acc=saved.get('best_val_acc',float('-inf'))
        # 同理 best_val_loss 用 inf：老断点没有这个字段，第一轮的 valid loss 一定是"新低"，
        # 不会一上来就背着 8 轮的坏计数被判早停
        best_val_loss=saved.get('best_val_loss',float('inf'))
        best_loss_epoch=saved.get('best_loss_epoch',0)
        bad_epochs=saved.get('bad_epochs',0)
        EARLY_STOP_PATIENCE=saved.get('early_stop_patience',EARLY_STOP_PATIENCE)
        if legacy:
            print("[resume] 这是旧版断点（只有模型+优化器+轮次）：")
            print("[resume]   续跑后第一轮会把 best_acc / best_loss 重新记一遍，之后照常")
            print("[resume]   该断点只存了'历史最优那轮'的模型，等于从那个最高分的位置继续，"
                  "不是断点前最后那轮")
        # 配置核对：断点里存了当时的 batch_size / 词表大小 / 数据条数。
        # 这份数据用固定 seed 切分，条数一样时切出来的训练集就是同一份，续跑才有意义。
        # ⚠️ 放在"有没有轮次可跑"的判断之前：配置换了却因为没轮次可跑而一个字不报，最坑
        mismatch=check_resume_config(saved,config,batch_size)
        if mismatch:
            print("⚠️ [resume] 断点与本次配置对不上：" + "，".join(mismatch))
            print("⚠️ [resume] 数据或词表变了等于换了一道题，接着用旧参数没有意义 —— 想重训请加 --fresh")
        if start_epoch>=epoch:
            print(f"[resume] 断点已经跑完 {start_epoch} 轮，而当前轮数上限是 {epoch} 轮，没有可跑的轮次。"
                  f"要接着多练几轮就调大轮数（例如 --epochs {start_epoch+10}）")

    session_time=0.0          # 本次进程内的用时，跟历史累计分开记，免得"本轮用时"在续跑后看着像没跑
    session_epochs=0
    completed_epochs=start_epoch     # 最后真正跑完的轮数
    early_stopped=False
    for epoch_i in range(start_epoch,epoch):
        completed_epochs=epoch_i+1
        # 这一轮之后就没有下一轮了：说明循环会正常走完，收尾时按"跑满"算。
        # 用"这一轮是不是最后一轮"来判，比循环跑完再回头看更省事，
        # 也让最后一次落盘的文件名/状态一次写对，不用事后再补一刀
        last_epoch_of_run=(epoch_i==epoch-1)
        start_t=time.time()   # 本轮计时起点放在训练开始前：本轮"训练+验证"的耗时都算进去
        # 每个 epoch 开始前打一次当前学习率：scheduler 改的就是 Adam.param_groups 里的 lr，
        # 直接读它才是优化器本轮实际要用的值，降没降一眼可见
        print(f"[train] epoch {epoch_i+1} 开始 | 当前学习率 lr={Adam.param_groups[0]['lr']:.6g}"
              + (f" | 混合精度 {str(amp_dtype).replace('torch.','')}" if use_amp else ""))
        train_loss,train_acc=train_one_epoch(model,train_dataloader,epoch_i,Adam,loss_function,
                                             clip,pr_ci,scaler,amp_dtype)
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
        session_time+=e_time
        session_epochs+=1
        # 平均轮时长用"已完成的总轮数"除：续跑时 total_train_time 里含着断点之前那些轮，
        # 只除本次轮数会把平均时长算大好几倍，剩余预估跟着离谱
        avg_time=total_train_time/(epoch_i+1)
        eta=avg_time*(epoch-epoch_i-1)          # 预估剩余 = 平均每轮 × 剩余轮数
        print(f"[time] 本轮用时 {_dur(e_time)} | 本次进程 {session_time/60:.2f} min/{session_epochs} 轮"
              f" | 累计 {total_train_time/60:.2f} min | 预计剩余 {_dur(eta)}"
              f" | 预计总时长 {_dur(avg_time*epoch)}")
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
                # 早停也算一种收尾：先把这一轮的状态落盘再 break。
                # 不落的话停在早停这一轮，续跑只能回到上一轮的 last.pth，那一轮白跑
                _save_state(model,Adam,scheduler,epoch_i+1,valid_acc,
                            best_val_acc,best_val_loss,best_loss_epoch,bad_epochs,
                            EARLY_STOP_PATIENCE,'finished',total_train_time,epoch,config)
                early_stopped=True
                break
        # 每跑完一轮覆盖写一次 last.pth：训练被 Ctrl+C / 掉线 / OOM 打断时，
        # 最坏情况只丢当前这一轮，不用从头再来。
        # 状态名分两种：跑满/早停是 finished（有始有终），否则就是 interrupted
        # （还打算接着跑，只是进程死了）。续跑时那句提示就是照这个状态说的
        status='finished' if (last_epoch_of_run or early_stopped) else 'interrupted'
        _save_state(model,Adam,scheduler,epoch_i+1,valid_acc,
                    best_val_acc,best_val_loss,best_loss_epoch,bad_epochs,
                    EARLY_STOP_PATIENCE,status,total_train_time,epoch,config)
    else:
        # for-else：循环跑满、没被早停 break 才走这里。
        # 正常单次训练到这里已经没什么可写的了（上面最后一次落盘就是 finished + 满轮数），
        # 只有"某一轮跑到一半进程死了、重启后续跑又发现没有剩余轮次"这种极端情况，
        # 断点里可能还是 interrupted —— 那时补一次收尾，别的什么都不做
        if session_epochs==0 and completed_epochs>start_epoch:
            _save_state(model,Adam,scheduler,completed_epochs,locals().get('valid_acc',0.0),
                        best_val_acc,best_val_loss,best_loss_epoch,bad_epochs,
                        EARLY_STOP_PATIENCE,'finished',total_train_time,epoch,config)
    # 收尾：不 close 的话最后一段数据可能还留在写缓冲区里没落盘
    if WRITER is not None:
        WRITER.close()

def _save_state(model,Adam,scheduler,done_epoch,val_acc,best_val_acc,best_val_loss,
                best_loss_epoch,bad_epochs,patience,status,total_train_time,epochs_planned,
                config=None):
    """save_checkpoint 的薄封装：把这一轮的状态凑齐后写 last.pth"""
    state={
        'best_val_acc':best_val_acc,
        'best_val_loss':best_val_loss,
        'best_loss_epoch':best_loss_epoch,
        'bad_epochs':bad_epochs,
        'early_stop_patience':patience,
        'status':status,
        'total_train_time':total_train_time,
        'epochs_planned':epochs_planned,
        # 配置走参数传进来，不读模块级 CONFIG：读全局时"main() 里填了 CONFIG"和
        # "train() 收到的是哪个 config"会各说各话，断点里就可能存下一份对不上的 meta
        # （自检里就真撞上过：传了 config，存进去的全是 None）
        **(config or {}),
    }
    save_checkpoint(LAST_MODEL_PATH,model,Adam,scheduler,done_epoch,val_acc,state)

def main():
    import torch
    from my_tokenizer import tokenize
    global log_root, CHECKPOINT_DIR, runs_root, BEST_MODEL_PATH, LAST_MODEL_PATH, LOG_CSV,src_tokens,tgt_tokens,device,WRITER,CONFIG
    args=parse_args()
    # ---- 训练超参（默认值按 24G 显存的 4090 调的）----
    # RL 从 1e-4 提到 2e-4：batch_size 从 128 提到 512 是 4 倍，按线性缩放该到 4e-4。
    # 但这个模型对 lr 敏感（注释里记着 lr 被砍到 6.25e-6 后 60 轮空转），
    # 折中取 2e-4，先跑一轮看 Grad/norm 有没有异常放大再决定要不要继续加
    RL=0.0002
    clip=1.0
    pr_ci=5#一个epoch打印多少中间信息
    # ✅ 先看要不要续跑：续跑就落回它自己那一轮的日志目录，TensorBoard 曲线接着画
    resume_path,resume_log_dir=resolve_resume(args)
    if resume_path is not None:
        print(f"[resume] 已找到断点：{resume_path}")
        log_root=resume_log_dir
    else:
        # 全新训练：日志目录名带时间戳，各跑各的不会互相覆盖
        log_root = "base_trian_log_" + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    os.makedirs(log_root, exist_ok=True)
    # ✅ 根据 log_root 设置其他路径
    CHECKPOINT_DIR = os.path.join(log_root, "checkpoints")
    # torch.save 不会自动创建父目录：只拼路径不建目录，第一次保存 best 就会 RuntimeError
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # TensorBoard 事件目录跟 checkpoints 并排，都在这一轮的 log_root 下
    WRITER = SummaryWriter(log_dir=os.path.join(log_root, "tb"))
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet18_9.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last.pth")
    # 调度器参数
    LR_FACTOR = 0.5
    # 5 → 12：上次 patience=5 时 lr 在 epoch 41 前后就被砍到 6.25e-6 冻结，后面 60 个 epoch 基本空转
    LR_PATIENCE = 12
    # ---- 配置：自己复现就自己写全,不依赖 utils ----
    # 路径写法跟 my_dictionary.py 的 __main__ 保持一致:写死绝对路径,看得见读的是哪份文件
    data_path=(args.data_path or
               r"/home/AI_agent/lll2/my_/train_translation/dataset/wmt_zh_en_training_corpus.csv")

    # ---- 训练规模：1000 万对（4090 上显存够，别再用 100 万对）----
    # 全语料 2475 万对 / 6.36GB。之前只取 100 万对，等于把最有效的抗过拟合手段废掉了：
    #   100 万对：训练 token 18.2M，模型 120M 参数 -> token/参数 0.15（严重过拟合区）
    #   1000 万对：训练 token 182M -> token/参数 2.96（进入可训练区间）
    # 实测超长丢弃率只有 0.03%（采样 20 万行统计），所以这个限流基本不丢数据
    max_pairs=10000000
    # [实测 2026-09-24] 20万→199932对(丢68)  100万→999548对  200万→1970365对
    # 数据量是压过拟合最有效的一招：run.log 那轮只用了 20 万对(占全语料 0.8%)就过拟合
    if args.max_pairs is not None:
        max_pairs=args.max_pairs      # 冒烟测试用：20000 对几分钟就能建完词表、跑完一轮
    max_sent_len=50#句子最多多少 token;也决定 tokenize 后的序列长度(=它+2)
    # ⚠️ 实测句长分布（采样 20 万对）：中文 mean 20.7 / p95 31 / p99 38 / max 73
    #    50 这个阈值只丢掉 0.03% 的句对，已经很贴边了，别再往下调 ——
    #    降到 30 要丢 8.87%，那是拿数据换序列长度，不划算
    #
    # min_count 2 → 3：词表从 zh 181564 / en 119110 压到约 11 万 / 7.9 万，
    # 参数从 120.5M 掉到 81.6M（tying 后 61.3M）。代价是 UNK 率从 1.84%/1.05%
    # 涨到 2.90%/1.63% —— 多 1 个点的 UNK 换掉一半词表参数，很划算：
    # 那些只出现 1~2 次的词占词表一多半，却只贡献不到 1% 的 token
    min_count=3#出现次数低于此值的词不单独占编号,归入 UNK
    # batch_size 128 → 512：4090 24G 显存放得开。原来卡在 128 是因为
    # "batch 20 和 128 每步耗时一样"——那是 CPU 侧拼装的瓶颈，不是显卡的。
    # y_p 那个 (B, L, V) 张量在 B=512/L=51/V≈79000 下约 3.8GB(fp32)，
    # 开 bf16 减半到 1.9GB，加上模型自身约 1GB，总共 3GB 出头
    batch_size=512
    num_workers=8#DataLoader 子进程数：0 = 主进程串行（原来的行为）
    # pin_memory：把 batch 放在"锁页内存"里，配合 _to_batch 里的 non_blocking=True
    # 能做到 H2D 拷贝和计算重叠。只有 CUDA 上才有意义，CPU 训练时自动关掉
    pin_memory=True
    amp=True     # 混合精度：bf16。4090 是 Ada，bf16 原生支持，不用 fp16+GradScaler 那套
    if args.batch_size is not None:
        batch_size=args.batch_size
    if args.min_count is not None:
        min_count=args.min_count
    if args.num_workers is not None:
        num_workers=args.num_workers
    if args.lr is not None:
        RL=args.lr
    amp=bool(args.amp) and amp

    # 512/16/2048/14+14 → 256/8/1024/7+7（撤掉翻倍）：
    # ① 与上次 33.7% 的基线同构，数据量实验才有可比性；② 翻倍版估 ~3.6~4GB，4GB 卡上大概率 OOM；
    # ③ 过拟合已确认（train 43.7% vs valid 33.7%），扩容只会放大它。要扩也是"先扩数据"，不是先扩模型
    #
    # ⚠️ 层数和 model_dim 这次一律不动。原因：打开 tying 之后总参数 61.3M，
    #    其中 Transformer 主体只有 12.86M（占 21%），剩下全在词表上。
    #    砍层数/d_model 省不了多少参数，却实打实削弱表达能力 —— 要压过拟合应该加数据
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
    # 混合精度的 dtype：只有真在 CUDA 上才开。
    # bf16 的动态范围和 fp32 一样，不会下溢，所以不需要损失缩放；
    # 老的卡（图灵及以前）没有 bf16，那种情况用 --no-amp 退回 fp32 更省事
    amp_dtype=None
    if amp and device.type=='cuda':
        amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pin_memory=pin_memory and device.type=='cuda'
    epoch=30#上限，不是计划：valid loss 见拐点后 8 轮会被早停接走（run.log 拐点在 epoch 29）
            # 断点续跑想多练几轮就传 --epochs，比改脚本省事

    if args.epochs is not None:
        epoch=args.epochs

    # ---- 1. 词表：有缓存直接加载,没有就现建并落盘 ----
    # 缓存文件名里带着 (max_pairs, max_sent_len, min_count),
    # 这三个参数一变文件名就变,会自动重建,不会静默用旧词表
    # ⚠️ 这次 max_pairs 和 min_count 都变了，缓存名也跟着变，会重新建表：
    #    1000 万对建词表 + tokenize 需要几分钟到十几分钟（100 万对实测准备 50s），
    #    这是正常的，不是卡住了
    print(f"[config] max_pairs={max_pairs:,} | min_count={min_count} | max_sent_len={max_sent_len}"
          f" | batch_size={batch_size} | lr={RL:.6g} | epochs<={epoch}"
          f" | num_workers={num_workers} | amp={amp_dtype if amp_dtype else 'off'}"
          f" | device={device}")
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

    # 断点里要记的训练配置：续跑时拿它对一遍，数据/词表换了就报警（见 train() 里的核对）
    CONFIG={
        'batch_size':batch_size,
        'max_sent_len':max_sent_len,
        'min_count':min_count,
        'max_pairs':max_pairs,
        'src_vocab_size':src_vocab_size,
        'y_vocab_size':y_vocab_size,
        'num_pairs':len(src_tokens),
    }

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

    # tying 之后模型参数 61.3M（词表 zh≈11万 / en≈7.9万），打一份出来对账：
    # 源嵌入 28.2M + 目标嵌入 20.2M + 主体 12.9M，输出层不再单独占参数
    print(f"[model] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
          f" | 词表 zh={src_vocab_size:,} en={y_vocab_size:,}"
          f" | 句对 {len(src_tokens):,}")

    train(model,epoch,batch_size,my_dictionary.PAD_TOKEN,RL,clip,pr_ci,LR_FACTOR,LR_PATIENCE,
          resume_path=resume_path,config=CONFIG,
          num_workers=num_workers,pin_memory=pin_memory,amp_dtype=amp_dtype)
    pass
if __name__=="__main__":
    main()
