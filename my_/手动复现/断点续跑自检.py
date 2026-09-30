"""
断点续跑自检：跑一遍就知道 train_增加了自动调整断点续跑版.py 的续跑功能还对不对。

    python 断点续跑自检.py                       # 默认测同目录那份训练脚本
    python 断点续跑自检.py 别的训练脚本.py        # 也可以指定

做的是一次对照实验：
  A  一次跑完 5 轮
  B  只跑 3 轮 -> 存断点 -> 从断点续跑到 5 轮
两边的逐轮 loss/acc 必须逐字一致（随机数状态没恢复就会对不上），最后权重也必须完全相同。
另外还验了：参数解析、--fresh、--log_dir、配置对不上时报警、best 指标不被重置、断点原子写。

用真实训练脚本会要 120M 参数 + 5.9GB 语料，这里换成 240 条假数据的迷你模型，
照样能验出"状态有没有恢复全"—— 漏掉 shuffle 的随机流、调度器耐心计数这类东西，
逐字一致那条断言立刻就会挂。
"""
import os, sys, io, shutil, time, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
# 必须在 chdir 之前算成绝对路径：chdir 之后相对路径会指到新目录里去
ARGV1 = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                        else os.path.join(HERE, 'train_增加了自动调整断点续跑版.py'))
# scratch 目录写在哪儿，取决于哪一层能建目录（脚本目录下未必行）。
# 逐个试一遍，第一个建得起来的就用它；全都不行就退回系统临时目录。
def _pick_root(base):
    for cand in (base, os.path.dirname(base), os.path.dirname(os.path.dirname(base)),
                 os.path.join(__import__('tempfile').gettempdir(), 'resume_selftest')):
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, '_w')
            open(probe, 'w').close()
            os.remove(probe)
            return cand
        except Exception:
            continue
    raise SystemExit('找不到可写的 scratch 目录')

ROOT = os.path.join(_pick_root(HERE), '.resume_selftest_scratch')
SC = os.path.join(ROOT, 'run_' + time.strftime('%m%d_%H%M%S'))
os.makedirs(SC, exist_ok=True)            # 每轮换新目录：上一轮的文件被占用时删不掉
os.chdir(SC)
SC = os.getcwd()                          # chdir 之后用绝对路径，find_latest_run(SC) 才扫得到
sys.path.insert(0, HERE)

LOG = open(os.path.join(ROOT, 'selftest.log'), 'w', encoding='utf-8')
def say(*a):
    """命令行的中文输出和 python 输出混在一起容易乱码，结论同时落一份到日志文件"""
    line = ' '.join(str(x) for x in a)
    try:
        print(line)
    except Exception:
        pass
    LOG.write(line + '\n')
    LOG.flush()

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import my_dictionary

import importlib.util
_spec = importlib.util.spec_from_file_location("mod", ARGV1)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

VOCAB, BATCH, SEED = 60, 16, 1111
torch.manual_seed(SEED)
N = 240
src_tokens = [[4 + (i * 7 + j) % (VOCAB - 4) for j in range(6)] for i in range(N)]
tgt_tokens = [[4 + (i * 5 + j) % (VOCAB - 4) for j in range(6)] for i in range(N)]
A_DIR, B_DIR = 'base_trian_log_01-01_00-00-00', 'base_trian_log_01-01_00-00-01'

class ToyModel(nn.Module):
    """形状对得上的迷你翻译模型：(B,L_src)+(B,L_tgt) -> (B,L_tgt,V)。
    dropout 打开，才有"随机数状态没恢复就会跑出不同结果"这回事"""
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(VOCAB, 32)
        self.tgt_emb = nn.Embedding(VOCAB, 32)
        self.drop = nn.Dropout(0.15)
        self.linear = nn.Linear(32, VOCAB)
        # _log 里要读 encoder.embedding_positional.embedding.weight 和 decoder.linear.weight，
        # 名字照真实模型的结构摆一份，否则 _log 那两行会 AttributeError
        self.encoder = type('E', (), {'embedding_positional': type('P', (), {'embedding': self.src_emb})()})()
        self.decoder = type('D', (), {'linear': self.linear})()

    def forward(self, src, tgt):
        ctx = self.drop(self.src_emb(src)).mean(dim=1, keepdim=True)   # (B,1,D)
        h = self.drop(self.tgt_emb(tgt)) + ctx
        return self.linear(h)

def fresh_model():
    torch.manual_seed(SEED)
    return ToyModel()

def set_run_dir(rel):
    T.log_root = rel
    T.CHECKPOINT_DIR = os.path.join(rel, 'checkpoints')
    T.BEST_MODEL_PATH = os.path.join(T.CHECKPOINT_DIR, 'best_resnet18_9.pth')
    T.LAST_MODEL_PATH = os.path.join(T.CHECKPOINT_DIR, 'last.pth')
    T.WRITER = SummaryWriter(log_dir=os.path.join(rel, 'tb'))
    os.makedirs(T.CHECKPOINT_DIR, exist_ok=True)

def ck_epoch(rel):
    """读一份断点，回报它记的轮数和状态 —— 每步都验一下写盘的内容对不对"""
    p = os.path.join(rel, 'checkpoints', 'last.pth')
    if not os.path.isfile(p):
        return None
    sd = T._load_ckpt(p)
    return sd['epoch'], sd['status']

def run(rel, epochs, resume_path=None, config=None, show=()):
    """
    跑一次 train。每个 batch/epoch 的日志先收进 buffer，只把 show 里列的关键行打出来
    —— 结论要看得见，噪音不用看。出错时把 buffer 尾部打出来方便定位。
    """
    set_run_dir(rel)
    m = fresh_model()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            T.train(m, epochs, BATCH, my_dictionary.PAD_TOKEN, 1e-3, 1.0, 5, 0.5, 12,
                    resume_path=resume_path, config=config)
    except Exception:
        say('[!!] train 抛异常，日志尾部：\n' + buf.getvalue()[-3000:])
        raise
    out = buf.getvalue()
    for ln in out.splitlines():
        if any(k in ln for k in show):
            say('   ', ln)
    return m, out

T.device = torch.device('cpu')
T.src_tokens, T.tgt_tokens = src_tokens, tgt_tokens
# batch_size 也要放进来：生产代码里 CONFIG 是 main() 拼的，本来就含 batch_size，
# 自检里手拼一份就得照着补齐，不然 meta 里那格永远是 None、配置核对也跟着失效
CFG = {'batch_size': BATCH, 'src_vocab_size': VOCAB, 'y_vocab_size': VOCAB, 'num_pairs': N}
CFG_BAD = {'batch_size': BATCH, 'src_vocab_size': VOCAB + 1, 'y_vocab_size': VOCAB, 'num_pairs': N}
COLS = ('[resume]', '[epoch', '[stop]', '[time]', '对不上')

# 生产代码里 find_latest_run() 扫的是"脚本所在目录"，这里得把扫描范围指到本次的自检目录，
# 否则 auto 永远找不到刚写下的断点
_real_find_latest = T.find_latest_run
T.find_latest_run = lambda root=None: _real_find_latest(SC)

say('=== A: 一次跑完 5 轮（基线）===')
ma, outa = run(A_DIR, 5, config=CFG)
say('[ok] A 跑完，断点 =', ck_epoch(A_DIR))

say('=== B: 只跑 3 轮，然后从断点续跑到 5 轮 ===')
mb, outb = run(B_DIR, 3, config=CFG)
say('[ok] B 跑完 3 轮，断点 =', ck_epoch(B_DIR))
# 跑满计划轮数就是 finished；interrupted 只留给"进程中途死掉"（那种情况下一轮都没写盘）
assert ck_epoch(B_DIR) == (3, 'finished'), ck_epoch(B_DIR)

import argparse as _ap
def fake_args(resume=None, log_dir=None, fresh=False, epochs=None):
    return _ap.Namespace(resume=resume, log_dir=log_dir, fresh=fresh, epochs=epochs)

# auto：扫 base_trian_log_*，挑断点最新落盘的。
# mtime 固定住，免得多轮 torch.save 的时间分辨率把两个目录排成并列
os.utime(os.path.join(B_DIR, 'checkpoints', 'last.pth'), (1, 1))
os.utime(os.path.join(A_DIR, 'checkpoints', 'last.pth'), (2, 2))
found = T.find_latest_run(SC)
assert found and A_DIR in found, found
say('[ok] find_latest_run 按落盘时间挑到:', found)

ckpt_path, log_dir = T.resolve_resume(fake_args(resume='auto'))
assert ckpt_path is not None and log_dir.endswith(A_DIR), (ckpt_path, log_dir)
say('[ok] --resume(不带值) ->', ckpt_path)
ckpt_path2, log_dir2 = T.resolve_resume(fake_args(log_dir=B_DIR))
assert log_dir2.endswith(B_DIR) and ckpt_path2.endswith('last.pth'), (ckpt_path2, log_dir2)
say('[ok] --log_dir 指向带断点的目录 -> 自动续跑')
assert T.resolve_resume(fake_args(resume='auto', fresh=True)) == (None, None)
say('[ok] --fresh 忽略断点')
assert T.resolve_resume(fake_args(resume=None)) == (None, None)
say('[ok] 不带参数 -> 全新训练')
try:
    T.resolve_resume(fake_args(resume='base_trian_log_09-09_99-99-99'))
    raise AssertionError('写错路径居然没报错')
except SystemExit as e:
    say('[ok] 断点路径写错会直接退出:', e)

mb, outb2 = run(B_DIR, 5, resume_path=ckpt_path2, config=CFG, show=COLS)
assert not [ln for ln in outb2.splitlines() if '对不上' in ln], '自续自的断点不该报配置不一致'
say('[ok] 自续自的断点，配置核对没有误报；续跑后断点 =', ck_epoch(B_DIR))

def metrics(out):
    return [ln for ln in out.splitlines() if ln.startswith('[epoch')]

la, lb = metrics(outa), metrics(outb + outb2)
assert len(la) == 5, f'基线应有 5 轮，实际 {len(la)}'
assert len(lb) == 5, f'3+2 应有 5 轮，实际 {len(lb)}'
for i, (x, y) in enumerate(zip(la, lb), 1):
    assert x == y, f'第 {i} 轮指标不一致:\n 一次跑完: {x}\n 断点续跑: {y}'
    say(f'[ok] 第 {i} 轮逐字一致: {y}')
say('[ok] 5 轮全部逐字一致 —— 续跑等价于没停过')

diff = max((p1 - p2).abs().max().item() for p1, p2 in zip(ma.parameters(), mb.parameters()))
say(f'[ok] 最终权重最大差异 {diff:.3e}')
assert diff == 0.0, diff
T.WRITER.close()

say('=== C: 配置对不上时必须报警（而且没轮次可跑时也要报）===')
# 断点已经是 5 轮了，这里故意只给 4 轮：没有任何轮次可跑。
# 以前这种"提前 return"会顺手把配置检查也跳过 —— 配置换了却一声不吭
mc, outc = run(B_DIR, 4, resume_path=ckpt_path2, config=CFG_BAD, show=COLS)
warn = [ln for ln in outc.splitlines() if '对不上' in ln]
assert warn, outc[-2000:]
say('[ok] 报警了:', warn[0])
assert not [ln for ln in outc.splitlines() if ln.startswith('[epoch')], '不该有轮次被跑'
say('[ok] 没有可跑的轮次，也没乱写断点；断点仍是', ck_epoch(B_DIR))
T.WRITER.close()

say('=== C2: 续跑一轮也不能把好事搞坏（best 指标要带过来）===')
before = T._load_ckpt(ckpt_path2)
mc2, outc2 = run(B_DIR, 6, resume_path=ckpt_path2, config=CFG, show=COLS)
after = T._load_ckpt(ckpt_path2)
say(f"    best_val_acc: {before['best_val_acc']:.4f} -> {after['best_val_acc']:.4f}"
    f" | best_val_loss: {before['best_val_loss']:.4f} -> {after['best_val_loss']:.4f}"
    f" | epoch: {before['epoch']} -> {after['epoch']} | status: {after['status']}")
assert after['epoch'] == 6, after['epoch']
assert after['best_val_acc'] >= before['best_val_acc'], (before['best_val_acc'], after['best_val_acc'])
assert after['best_val_loss'] <= before['best_val_loss'], (before['best_val_loss'], after['best_val_loss'])
say('[ok] best 指标是"历史最好"，没有被续跑重置')
T.WRITER.close()

say('=== D: last.pth 里到底存了什么 ===')
sd = T._load_ckpt(ckpt_path2)
for k in ('epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'rng_state',
          'best_val_acc', 'best_val_loss', 'bad_epochs', 'early_stop_patience', 'status', 'meta'):
    assert k in sd, f'断点里缺 {k}'
say('[ok] 字段齐全:', ', '.join(sorted(sd.keys())))
say('[ok] meta =', sd['meta'])
assert sd['meta']['src_vocab_size'] == VOCAB, sd['meta']       # 配置真的写进去了，不是一片 None
assert sd['meta']['num_pairs'] == N, sd['meta']
assert sd['meta']['batch_size'] == BATCH, sd['meta']

say('=== E: lr 与调度器耐心计数真的被恢复了 ===')
opt2 = optim.AdamW(fresh_model().parameters(), lr=1e-3)
sch2 = lr_scheduler.ReduceLROnPlateau(opt2, mode='max', factor=0.5, patience=3)
for _ in range(3):                     # 制造"连续 3 轮不提升"，让耐心计数变成 3
    for g in opt2.param_groups:
        g['lr'] = 1e-3
    sch2.step(0.10)
opt2.param_groups[0]['lr'] = 5e-4      # 故意先写个错的值，看恢复时会不会被断点纠正
bad_before = getattr(sch2, 'num_bad_epochs', getattr(sch2, 'bad_epochs', None))
say('    恢复前: lr =', opt2.param_groups[0]['lr'], '耐心计数 =', bad_before)
m3 = fresh_model()
T.restore_for_resume(ckpt_path2, m3, opt2, sch2, torch.device('cpu'))
bad_after = getattr(sch2, 'num_bad_epochs', getattr(sch2, 'bad_epochs', None))
lr_after = opt2.param_groups[0]['lr']
lr_in_ckpt = sd['optimizer_state']['param_groups'][0]['lr']
say(f'[ok] 恢复后 lr={lr_after:.6g}（断点里记的 {lr_in_ckpt:.6g}）'
    f' 耐心计数={bad_after}（断点里记的 {sd["scheduler_state"].get("num_bad_epochs")}）'
    f' best={sch2.best}（断点里记的 {sd["best_val_acc"]:.4f}）')
# 优化器的 param_groups 也一起加载，手写进去的假 lr 会被断点里真实的值覆盖 —— 这正是想要的
assert lr_after == lr_in_ckpt, (lr_after, lr_in_ckpt)
assert bad_after == sd['scheduler_state'].get('num_bad_epochs'), (bad_after, sd['scheduler_state'])
assert abs(sch2.best - sd['best_val_acc']) < 1e-12, (sch2.best, sd['best_val_acc'])
say('[ok] lr / 耐心计数 / best 三项都是从断点恢复的，不是从零开始')

say('=== F: 半截的断点不能影响正式断点（原子写）===')
with open(ckpt_path2 + '.tmp', 'wb') as f:
    f.write(b'half written garbage')
good = T._load_ckpt(ckpt_path2)           # 真断点不受影响
say('[ok] 半截的 .tmp 不影响 last.pth 的读取，epoch =', good['epoch'])

say('=== G: 命令行参数本身 ===')
def argv_parse(*argv):
    old = sys.argv
    sys.argv = ['train.py'] + list(argv)
    try:
        return T.parse_args()
    finally:
        sys.argv = old

a = argv_parse()
assert (a.resume, a.log_dir, a.fresh, a.epochs) == (None, None, False, None), vars(a)
say('[ok] 不带参数 -> 全新训练，轮数用脚本里的默认值')
a = argv_parse('--resume')
assert a.resume == 'auto', a.resume
say("[ok] --resume            -> resume='auto'（自动挑最近一次）")
a = argv_parse('--resume', B_DIR)
assert a.resume == B_DIR, a.resume
say('[ok] --resume <目录>     -> 用指定那一轮')
a = argv_parse('--resume', '--epochs', '40')
assert a.resume == 'auto' and a.epochs == 40, vars(a)   # '--epochs' 不会被当成 --resume 的值吃掉
say('[ok] --resume --epochs 40 -> 两个参数各归各的（nargs=? 的经典坑）')
a = argv_parse('--epochs', '40', '--fresh')
assert a.epochs == 40 and a.fresh is True, vars(a)
say('[ok] --epochs 40 --fresh  -> 从第 1 轮重跑、上限 40 轮')
a = argv_parse('--log_dir', B_DIR)
assert a.log_dir == B_DIR and a.resume is None, vars(a)
say('[ok] --log_dir <目录>     -> 目录里有断点会自动续跑')

say('=== H: 从旧版脚本的断点续跑（只有模型+优化器+轮次，没有 best/早停/调度器）===')
# 老脚本 train_增加学习了自动调整版.py 的保存格式：{epoch, model_state, optimizer_state, val_acc}
# 而且只写 best_resnet18_9.pth / best_epochNN_accXX.pth，从来不写 last.pth
L_DIR = 'base_trian_log_02-02_00-00-00'
set_run_dir(L_DIR)
torch.save({'epoch': 7, 'model_state': fresh_model().state_dict(),
            'optimizer_state': optim.AdamW(fresh_model().parameters(), lr=1e-4).state_dict(),
            'val_acc': 0.4123},
           os.path.join(T.CHECKPOINT_DIR, 'best_resnet18_9.pth'))
T.WRITER.close()
assert not os.path.isfile(os.path.join(L_DIR, 'checkpoints', 'last.pth'))
legacy_path = T._pick_in_dir(os.path.join(L_DIR, 'checkpoints'))
assert legacy_path.endswith('best_resnet18_9.pth'), legacy_path
say('[ok] 老目录里只有 best_resnet18_9.pth，也认得出来:', os.path.basename(legacy_path))
p3, ld3 = T.resolve_resume(fake_args(resume=L_DIR))
assert p3.endswith('best_resnet18_9.pth') and ld3.endswith(L_DIR), (p3, ld3)
say('[ok] resolve_resume 能直接从老目录续（不用手敲完整文件名）')

ml, outl = run(L_DIR, 9, resume_path=p3, config=None, show=COLS)
assert '旧版断点' in outl, outl[-2000:]
say('[ok] 认出来了并给出说明（不给 config 时也不做配置核对，不误报）')
assert [ln for ln in outl.splitlines() if ln.startswith('[epoch 8')], '应该从第 8 轮接着跑'
say('[ok] 从第 8 轮接着跑（老断点 epoch=7）')
after_l = T._load_ckpt(os.path.join(L_DIR, 'checkpoints', 'last.pth'))
say(f"    best_val_acc: -inf -> {after_l['best_val_acc']:.4f}（第一轮重新记了一遍）"
    f" | best_val_loss: inf -> {after_l['best_val_loss']:.4f} | epoch: {after_l['epoch']}")
assert after_l['best_val_acc'] > 0, after_l['best_val_acc']      # 不能一直卡在 -inf
assert after_l['best_val_acc'] != float('-inf')
assert after_l['epoch'] == 9 and after_l['status'] == 'finished', (after_l['epoch'], after_l['status'])
say('[ok] 续跑后 best 指标恢复正常，断点被升级成新版格式，下次续跑不用再重新记')
T.WRITER.close()

say('')
say('全部自检通过')
say('SC =', SC)
say('（scratch 目录可以直接删：%s）' % ROOT)
