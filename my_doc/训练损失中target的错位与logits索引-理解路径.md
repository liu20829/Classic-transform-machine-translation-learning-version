# 训练损失中 target 的错位与 logits 索引

> 一句话：从"训练损失那两行是什么意思"出发，追踪一条 target 被切两刀、被摊两次的完整路径，最后校准一个想错了的词——被索引的不是独热编码，是 logits。
> 关键词：teacher forcing, target shift, contiguous, view, CrossEntropyLoss, logits
> 生成日期：2026-09-21

## 起点：这份文档要回答的问题链

| # | 当时问的 | 由什么触发 |
|---|---|---|
| ① | 这两行是什么意思？ | 读到 `train_ch_en.py` 训练循环里 loss 前那两行 |
| ② | `.view(-1)` 我理解这操作是什么意思？这不是直接给操作成一维了吗？ | ①的答案讲了 `(N,L,vocab) → (N*L,vocab)`，对"摊平"起了疑惑 |
| ③ | 用 target 的整数去索引独热编码，跟图像分类一样？ | ②的答案讲了"两边同步摊、一一对应不丢" |

**章节顺序即提问顺序。** 这不是教科书目录——每章对应一个当时真正卡住的地方，按卡住的先后排。①→②→③ 是一条因果链：①的答案里出现了"摊平"，才有了②；②的答案里出现了"一一对应"，才有了③。

---

## 一、这两行是什么意思？

**承接**：起点是本项目 `train_ch_en.py` 训练循环里的四行。前两行（拼 batch）一眼能懂，问题是后两行。

```python
# train_ch_en.py —— train() 循环体

# 把 batch 里 N 个样本堆成张量
input  = torch.cat(input,  dim=0).to(utils.device)   # (N, seq_len)
target = torch.cat(target, dim=0).to(utils.device)   # (N, seq_len)

# output -> (N, seq_len, tgt_vocab_size) ; target -> (N, seq_len)
output, _ = model(input, target[:,:-1])              # ← 先注意这个 [:, :-1]

# removing the first token of <SOS> and then flattening 2D to 1D tensor
output = output.contiguous().view(-1, output.shape[-1])
target = target[:,1:].contiguous().view(-1)          # ← 以及这两行
```

### 1.1 不看上一行，这两行读不懂

`model(input, target[:,:-1])` 这一行藏着整件事的前提：**喂给解码器的 target 已经被切过第一刀了**（去掉最后一位）。

所以完整的三刀是这样的：

| 序号 | 写法 | 动作 | 去哪 |
|---|---|---|---|
| 第 1 刀 | `target[:, :-1]` | 去掉**最后**一位 | 喂给解码器 |
| 第 2 刀 | `target[:, 1:]` | 去掉**第一**位 | 当损失函数的答案 |
| 第 3 刀 | `.view(...)` | 摊平 | 喂给 `CrossEntropyLoss` |

第 1、2 刀就是"**错开一位**"（shift by one）。

### 1.2 错开一位：一条 target 被切成两个错位的序列

取一条真实的 target（真实长度是 52，这里用 8 示意）：

```
下标:   0    1    2    3    4    5    6    7
token: SOS  我   爱   你  EOS  PAD  PAD  PAD
```

两刀砍出：

```
喂给解码器 [:, :-1] : SOS  我   爱   你  EOS  PAD  PAD      ← 去掉最后一位
要预测的  [:,  1:] : 我   爱   你  EOS  PAD  PAD  PAD      ← 去掉第一位(SOS)
```

长度都是 7，**位置却错开一格**。逐位置对齐看：

| 解码器第 i 步看到的 | 应当预测出的 |
|---|---|
| `SOS` | `我` |
| `SOS 我` | `爱` |
| `SOS 我 爱` | `你` |
| `SOS 我 爱 你` | `EOS` |

实测输出（用 `SOS=1, EOS=2, PAD=0` 跑一遍）：

```
原始 target        : ['SOS', 10, 11, 12, 'EOS', 'PAD', 'PAD', 'PAD']
喂给解码器 [:, :-1] : ['SOS', 10, 11, 12, 'EOS', 'PAD', 'PAD']
要预测的  [:, 1:]  : [10, 11, 12, 'EOS', 'PAD', 'PAD', 'PAD']
两者长度: 7 和 7

解码器第0步 看到 SOS    -> 应预测出 10
解码器第1步 看到 10     -> 应预测出 11
解码器第2步 看到 11     -> 应预测出 12
解码器第3步 看到 12     -> 应预测出 EOS
```

### 1.3 两刀为什么正好这么砍

不是凑出来的，是两头都堵死了：

- **`[:, :-1]` 去掉最后一位** —— 序列的最后一位没有"下一个词"可供预测。留着它，就有一个位置没有答案，对不上。
- **`[:, 1:]` 去掉第一位** —— 第一位是 `SOS`（起始符）。**没有人要预测 SOS**，它是人为塞进去的，不是语料里真实存在的词。留着它，就有一个答案没有输入，同样对不上。

所以只能"各去掉一头"，剩下来的正好长度相等、位置相差一格。

这正是 **teacher forcing**：每一步拿"到目前为⽌的正确句子"去预测下一个词。名字里的 "teacher" 就是指"用真实答案当老师喂进去"，而不是让模型吃自己上一步的输出。

### 1.4 挡在路上的 `.contiguous()`：切片是视图，内存不连续

写成 `target[:,1:].contiguous().view(-1)` 而不是直接 `target[:,1:].view(-1)`，原因在**切片不搬内存**。

`target[:, 1:]` 返回的是原张量的**视图**（view）——它没有复制数据，只是换了个"怎么看这块内存"的描述。而 `view()` 要求连续内存，直接调会报错。实测：

```python
import torch
tgt = torch.arange(16).reshape(2, 8)
print(tgt.shape, tgt.stride())        # (2, 8)  stride (8, 1)

sl = tgt[:, 1:]
print(sl.shape, sl.stride())          # (2, 7)  stride (8, 1)  ← 期望是 (7,1)
print(sl.is_contiguous())             # False

sl.view(-1)
# RuntimeError: view size is not compatible with input tensor's size and stride
# (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

print(sl.contiguous().view(-1).tolist())   # [1,2,3,4,5,6,7, 9,10,11,12,13,14,15] ✓
```

**怎么读 stride `(8, 1)`**：stride 是"沿这一维走一步，要在内存里跳几格"。

- 第 0 维跳 8 格 —— 因为原张量每行 8 个数，下一行的开头要跳过 8 个
- 第 1 维跳 1 格 —— 行内相邻

切片之后形状变成 `(2, 7)`，第 1 维只取 7 个数。但第 0 维的 stride **还是 8**（内存布局没变，第 1 行第 0 个元素仍然在第 8 格）。于是"逻辑上相邻的两行"在内存里隔着第 8 格那个**被切掉的元素**。

`view()` 无法表达这种"跳着取"的形状，所以拒绝执行。`.contiguous()` 先复制出一块紧凑的内存，再 `view()` 就顺理成章。

> ⚠️ 一个反直觉的坑：如果 batch 只有 1 行（`tgt` 形状 `(1, 8)`），`tgt[:,1:]` 的 **`is_contiguous()` 会返回 `True`**，`view(-1)` 也能直接跑通。原因是 PyTorch 的连续性判定**跳过长度为 1 的维度**——第 0 维长度为 1，它的 stride 是多少都无所谓。

### 1.5 白话翻译

一句话：**先把答案切出两份错开一格的序列，一份喂进去，一份留着对答案。**

用具体数字走一遍（target 是 `[SOS, 10, 11, 12, EOS, PAD, PAD, PAD]`，长度 8）：

1. **喂给解码器的**是前 7 个：`[SOS, 10, 11, 12, EOS, PAD, PAD]`
2. **要它预测的答案是**后 7 个：`[10, 11, 12, EOS, PAD, PAD, PAD]`
3. 第 0 步：模型只看到 `SOS`，它必须吐出 `10`
4. 第 1 步：模型看到 `SOS, 10`，它必须吐出 `11`
5. 第 2 步：模型看到 `SOS, 10, 11`，它必须吐出 `12`
6. 第 3 步：模型看到 `SOS, 10, 11, 12`，它必须吐出 `EOS`

**`contiguous` 这个词**在这里的意思是"内存里挨着"：`target` 原本 8 个数挨着排；切掉第一个之后，逻辑上要的 7 个数在内存里**不挨着**（第 8 格还夹在中间）。`view` 要求挨着才肯干活，所以先 `.contiguous()` 把要用的这 7 个数复制成一块挨着的新内存。

**`SOS` 为什么不算答案**：它是"开始"这个信号的编号，不是语料里出现过的词。翻译生成时它由程序塞进去，不会从模型嘴里吐出来，所以不参与预测。

---

## 二、`.view(-1)` 不是直接给操作成一维了吗？

**承接**：上一章末尾说了 `.view` 负责"摊平"。但摊平之后二维的形状 `(N, L)` 就没了——**这个结构丢了，答案还怎么跟预测对得上？** 这是这一章要解的疑惑。

### 2.1 摊平是"两边同步摊"，不是单方面压扁

关键在于：**output 和 target 用的是同一套规则在摊**。

```python
output = output.contiguous().view(-1, output.shape[-1])   # (N, L, vocab) -> (N*L, vocab)
target = target[:,1:].contiguous().view(-1)               # (N, L)        -> (N*L,)
```

用 2 行 3 列的实际数据看清楚"第 k 个元素原来在哪":

```python
import torch
N, L, V = 2, 3, 4
tgt = torch.tensor([[10,11,12],[20,21,22]])

flat = tgt.view(-1)
print(flat.tolist())      # [10, 11, 12, 20, 21, 22]

for k in range(N*L):
    i, j = k // L, k % L
    print('第%d个=%2d  <= 原[%d,%d]=%2d' % (k, flat[k], i, j, tgt[i,j]))
```

输出：

```
第0个=10  <= 原[0,0]=10
第1个=11  <= 原[0,1]=11
第2个=12  <= 原[0,2]=12
第3个=20  <= 原[1,0]=20
第4个=21  <= 原[1,1]=21
第5个=22  <= 原[1,2]=22
```

原来在第 `(i, j)` 的信息，摊平后落在第 `i*L + j` 个。**位置换了，内容一个没丢、顺序也没乱。**

而 output 那边用**同一个规则**摊：

```
output (N, L, vocab)  ->  (N*L, vocab)
target (N, L)         ->  (N*L,)
```

第 k 行 output 永远配第 k 个 target，**指的还是原来同一个位置**。配对关系一个没差。

### 2.2 为什么损失函数不需要那个二维结构

因为交叉熵是**逐个位置独立算的**：

| 位置 | 拿什么算 | 得到 |
|---|---|---|
| `(0,0)` | output 第 0 行的 vocab 个分数 vs 正确编号 `10` | 一个损失值 |
| `(0,1)` | output 第 1 行的 vocab 个分数 vs 正确编号 `11` | 一个损失值 |
| `(1,0)` | output 第 3 行的 vocab 个分数 vs 正确编号 `20` | 一个损失值 |
| ... | ... | ... |

`N × L` 个**互相之间毫无关系**的小分类问题。既然是独立的，"排成一队算"和"保持二维算"结果必然相同——**那个二维结构对计算没有任何用处**。

这也是为什么把 output 转置成 `(N, vocab, L)` 配 `(N, L)` 能算出**一模一样**的值：

```python
import torch, torch.nn as nn
N, L, V = 2, 51, 10
out = torch.randn(N, L, V)
tgt = torch.randint(0, V, (N, L))

print(nn.CrossEntropyLoss()(out.reshape(-1,V), tgt.reshape(-1)).item())   # 2.8799
print(nn.CrossEntropyLoss()(out.permute(0,2,1), tgt).item())             # 2.8799
```

换个排法，答案不变——结构确实是无关的。

### 2.3 `view(-1)` 里的 `-1` 是什么

`-1` 的意思是"**这一维你自己算**"。PyTorch 拿总元素数除以已知的维度：

```
target.view(-1)          总元素 N*L，只给了一个 -1  ->  N*L
output.view(-1, vocab)   总元素 N*L*vocab，已知 vocab -> 第一维 = N*L
```

等价于 `reshape`，但 `-1` 让形状跟着数据走，不用手算。

### 2.4 白话翻译

一句话：**把两个张量按同一个顺序"排成一队"，队里第 k 号永远对第 k 号。**

用具体数字走一遍（N=2, L=3，词表大小 V=4）：

1. `target` 本来是记账本，两行三列：`[[10,11,12],[20,21,22]]`
2. 摊平 = 撕掉行边界、一行读到底：`[10, 11, 12, 20, 21, 22]`，第 3 个元素是 `20`
3. `output` 本来是 `[[[...4个数...], [...], [...]], [[...], [...], [...]]]`——2 行 3 列，每格里 4 个数
4. 摊平 = 每格的 4 个数打包成一行，按同样顺序摞起来，得到 6 行 4 列
5. 于是"第 3 行 output"（4 个数）要预测出的编号就是"第 3 个 target"=`20`

**为什么不需要保留二维**：第 `(0,0)` 格和第 `(1,0)` 格的损失是分开算的、分开求和的，互相不看对方。既然互不相干，那"它们是在同一行还是不同行"就没意义了。

**`-1` 具体算了什么**：`target` 一共 6 个数，`view(-1)` 说"你算吧" → 6 除以 1 → 形状是 `(6,)`。`output` 一共 24 个数，`view(-1, 4)` 说"第二维是 4，第一维你算" → 24 ÷ 4 = 6 → 形状是 `(6, 4)`。

---

## 三、拿 target 的整数去索引，跟图像分类一样吗？

**承接**：上一章确定了"两边排成一队、第 k 号对第 k 号"。那么具体是怎么用这个 target 编号的？当时的理解是"**拿整数去索引独热编码**，跟图像分类一样"。这一章校准这句话里的一处偏差——但先说：**大框架是通的。**

### 3.1 "跟图像分类一样"——这个类比是官方文档明写的

```
图像分类:  output (N, 1000)   target (N,)      N 张图,每张在 1000 类里选一个
这里:      output (N*L, V)    target (N*L,)    N*L 个位置,每个位置在 V 个词里选一个
```

结构完全一致，只是把"每张图"换成了"每个位置"。`N × L` 个位置就是 `N × L` 个独立的分类样本。

这不是口头比喻——PyTorch 官方文档在 `CrossEntropyLoss` 的词条里就是这么推荐的：

> Input: Shape `(C)`, `(N,C)` or `(N,C,d1,d2,...,dK)` with K≥1 … **The last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.**
> — [CrossEntropyLoss — PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

"逐像素（per-pixel）算交叉熵损失，就像 2D 图像那样"——句中的 "per-pixel" 对应到本项目，就是 "per-position"（逐位置）。

### 3.2 转折点：被索引的不是独热编码

这一处是整条链上唯一想错的地方，值得单独成节。

原话是"用 target 的整数去索引 `(N*L, vocab)` 这个里面的**独热编码**"。**索引的机制描述对了，被索引的东西名字错了。**

`output` 里存的不是独热编码，而是 **logits**——未归一化的原始分数。它们的特征：

- 可以是**负数**
- 可以**任意大**
- **加起来不等于 1**

实测一行真实形态的 logits（V=5）：

```
logits(不是独热,是原始分数): [2.0, 1.0, 0.5, -1.0, 0.0]
```

对照独热编码长什么样——长度 V、只有一位是 1、其余全 0：

```
独热(正确词是 0 号):        [1.0, 0.0, 0.0,  0.0, 0.0]
```

两者长得完全不像。官方文档对此有明确措辞：

> The input is expected to contain the **unnormalized logits** for each class (which do not need to be positive or sum to 1, in general).

`logits` 要变成概率，得先过 softmax；而这个 softmax 是 `CrossEntropyLoss` **内部替你做的**，你不用自己加。

### 3.3 索引取值 vs 独热相乘：同一件事的两种算法

那"独热"这个印象从哪来？它来自一个**完全正确的等价算法**——只是 PyTorch 没这么实现。

从 logits 到损失，完整链条是：

```
output[row]        logits:V 个原始分数              [2.0, 1.0, 0.5, -1.0, 0.0]
      ↓ softmax
                  概率:非负、加起来等于 1
      ↓ 按 target 的编号取出那一个
                  正确词拿到的概率
      ↓ 取 -log
                  损失值
```

写成公式：

$$\text{loss} = -\log\Big(\text{softmax}(\text{logits})[k]\Big), \quad k = \text{target 的编号}$$

那个"取出来"的动作，有两种等价写法。实测三者数值：

```python
import torch, torch.nn as nn
logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, 0.0]])
target = torch.tensor([0])

# ① PyTorch 的做法
print(nn.CrossEntropyLoss()(logits, target).item())          # 0.574438

logp = torch.log_softmax(logits, dim=1)

# ② 索引取值:直接拿编号去取
print(-logp[0, target[0]].item())                             # 0.574438

# ③ 独热相乘:造一个独热向量,跟概率逐元素相乘再求和
onehot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
print(-(onehot * logp).sum().item())                          # 0.574438
```

**三者都是 `0.574438`** —— 数学上完全等价。

差别只在开销：

| 做法 | 步骤 | 每个位置的开销 |
|---|---|---|
| 独热相乘 | 造一个长度 V 的 `[0,0,1,0,...]`，与概率逐元素相乘，求和 | 多一个 **V 长**的向量 |
| **索引取值** | 直接 `概率[正确编号]` | 一次下标访问 |

本项目英文词表 `V = 57942`，一个 batch 有 `N × L` 个位置。若真按独热算，光这些零向量就白占一大片内存和带宽。**数学等价，所以没理由不用快的那个**——PyTorch 选的是索引取值。

### 3.4 白话翻译

一句话：**把每个位置当成一张待分类的图，模型给出一堆分数，损失函数只看"正确答案那一项"的分数有多低。**

用具体数字走一遍（V=5，正确词编号 k=0，logits = `[2.0, 1.0, 0.5, -1.0, 0.0]`）：

1. **原始分数**是 `[2.0, 1.0, 0.5, -1.0, 0.0]` —— 5 个数，有正有负，加起来是 2.5，不是 1
2. 过 softmax 变成**概率**：加起来等于 1，第 0 项（分数最高的那个）占比最大
3. **正确词是 0 号**，所以只取第 0 项那个概率
4. 取 `-log` 得到 `0.574438`
5. 换另一种算法——把目标写成独热 `[1,0,0,0,0]`，跟概率逐项相乘（只有第 0 项留下来，其余乘 0），再求和，得到 **同一个数** `0.574438`
6. 所以"直接按编号取那一项"和"乘一个独热再求和"是同一件事——PyTorch 用前者，因为**不用真去造那个长度 5（真实是 57942）的独热向量**

**"正确词拿到的概率越低，损失越大"**：如果正确词的分数很高（比如 5.0，别的是 0），softmax 后它概率接近 1，`-log(1) = 0`，损失接近 0；反过来它概率接近 0，`-log(0)` 趋向无穷大。模型的目标就是把正确词那一项的分数推高。

---

## 附、系统定义与对比

> 以下按需查阅，不属于问题链本身。

### A.1 teacher forcing 与 exposure bias

**teacher forcing** 指训练时把真实答案（ground truth）作为解码器的输入，而不是把模型上一步的输出喂回去。Transformer 原论文 "Attention Is All You Need" 没有用这个词，但它从始至终就是全程 teacher forcing——解码器的输入序列就是目标序列右移一位的版本。

**为什么可以整条一次性喂进去**：朴素描述是"一步喂一个词"，但因为解码器自注意力有因果掩码（causal mask），位置 $i$ 只能看见位置 $0 \dots i-1$，所以整条一起喂和逐词喂**在数学上等价**，却能并行计算。

**exposure bias（曝光偏差）**：训练时模型看到的是"永远正确的历史"，推理时看到的是"自己上一步吐出来的（可能错的）词"。两者分布不一致，一旦某步错了就会滚雪球。缓解手段有 scheduled sampling / curriculum learning（Bengio et al., 2015）等。

**推理阶段不一样**：推理时没有 target 序列，模型从只有起始符的空序列开始，每步把自己的输出接回去，直到吐出 `EOS`。

### A.2 `view` / `reshape` / `contiguous` 对照

| 方法 | 对非连续张量的行为 | 是否复制 |
|---|---|---|
| `.view(shape)` | **报错** | 不复制（所以要求连续） |
| `.reshape(shape)` | 能跑：先看能否返回视图，不行就复制 | 视情况 |
| `.contiguous()` | —— 本身不改变形状，只返回连续副本 | 是 |
| `.is_contiguous()` | —— 检查是否连续 | 否 |
| `.stride()` | —— 返回每一维的步长 | 否 |

**选择建议**：

- 明确知道内存连续、或想要"不复制"的性能保障 → `.view()`
- 只想改形状、不在意是否复制 → `.reshape()`（它把判断藏起来了，但也不用你操心）
- 本项目这种"切过片、一定要摊平"的场景 → `.contiguous().view()`，意图最直白

### A.3 `CrossEntropyLoss` 形状速查

| | 形状 | 说明 |
|---|---|---|
| **Input** | `(C)` / `(N,C)` / `(N,C,d1,...,dK)` | C 恒在 **dim 1**，是类别数 |
| **Target（类别编号）** | `()` / `(N)` / `(N,d1,...,dK)` | dtype 必须是 `long`，取值在 `[0,C)` |
| **Target（类别概率）** | 与 input 同形 | 概率版，需自己保证和为 1 |

**最关键的一条**：**类别数必须落在 dim 1**。这就是为什么本项目必须摊平:

```python
# 不摊平:类别数落在 dim 2,CrossEntropyLoss 会误把 L 当类别数
nn.CrossEntropyLoss()(out, tgt)      # out (N,L,V), tgt (N,L)
# RuntimeError: Expected target size [2, 10], got [2, 51]
#                                          ^^ 它以为类别是 V=10,期待 target 是 (2, 10)

# 摊平后
nn.CrossEntropyLoss()(out.reshape(-1,V), tgt.reshape(-1))   # ✓
```

**另一个等价写法**是把类别维转置到 dim 1（实测损失值相同）：

```python
nn.CrossEntropyLoss()(out.permute(0,2,1), tgt)   # (N,V,L) 配 (N,L)  ✓
```

**`ignore_index` 参数**：本项目用 `nn.CrossEntropyLoss(ignore_index=dict.PAD_TOKEN)`。

```python
# train_ch_en.py
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_TOKEN)
```

含义：target 中等 by 该编号的位置**不参与损失计算**。因为填充的 `PAD`（编号 0）是人为对齐长度塞进去的占位符，没有语义——不排除的话，模型会花大量容量去学"预测 PAD"，而这些位置的样本数量往往过半。

---

## 易混淆点汇总表

| 容易搞混的 | 正确的理解 | 出处 |
|---|---|---|
| 喂给解码器的和当答案的是同一条 target | 是**错开一位**的两条：`[:, :-1]` 喂进去，`[:, 1:]` 当答案 | 1.1 |
| `SOS` 也是要预测的词之一 | `SOS` 是人为塞的起始信号，不参与预测，所以答案要从 `[:, 1:]` 切掉它 | 1.3 |
| 切片 `.view(-1)` 可以直接调 | 切片是**视图**，内存不连续，`view` 会报错；必须先 `.contiguous()` | 1.4 |
| `is_contiguous()` 对 `(1, 8)` 切片后返回 `True` 说明切片一定连续 | 只是 batch=1 时第 0 维长度为 1，而连续性判定**跳过长度为 1 的维度** | 1.4 |
| 摊平成一维 = 把结构信息丢了 | 结构没丢：原来在 `(i,j)` 的信息落在第 `i*L+j` 个，且 output 用**同一规则**摊，第 k 号仍对第 k 号 | 2.1 |
| `view(-1, vocab)` 的第一维要自己算好填进去 | `-1` 就是"你自己算"，PyTorch 拿总元素数除以已知维度 | 2.3 |
| `output` 里存的是独热编码 | 是 **logits**：有正有负、可任意大、不归一化。softmax 是损失函数内部替你做的 | 3.2 |
| "用整数索引" 和 "独热相乘" 是两种不同算法 | 数学上**完全等价**（实测同为 `0.574438`），只是"索引取值"省掉一个 V 长的零向量 | 3.3 |
| 摊平只是写法喜好，不摊平也能跑 | **必须摊平**：`CrossEntropyLoss` 认定 dim 1 是类别维，本项目的类别维在最后 | A.3 |
| 交叉熵要自己先 softmax | 不用。`CrossEntropyLoss` = `log_softmax` + `nll_loss`，喂原始 logits 即可 | A.3 |

---

## 总结

1. **喂进去的 target 被切掉最后一位**（`[:, :-1]`）—— 因为最后一位没有"下一个词"可预测。
2. **当答案的 target 被切掉第一位**（`[:, 1:]`）—— 因为 `SOS` 是人为起始符，没有人要预测它。
3. **两刀砍完位置正好错开一格** —— 这才有了"每一步都在预测下一个词"的 teacher forcing 结构。
4. **`contiguous()` 是因为切片不搬内存** —— 切片是视图，stride 仍是 `(8,1)` 而非 `(7,1)`，`view` 拒绝执行。
5. **`view(-1)` 把两个张量按同一顺序排成一队** —— 原来在 `(i,j)` 的落在第 `i*L+j` 个，两边同步摊所以一一对应不丢。
6. **摊平是必需的，不是风格选择** —— `CrossEntropyLoss` 认定 dim 1 是类别数，而本项目的类别维在最后。
7. **被索引的是 logits 不是独热编码** —— 有正有负、不归一化；softmax 由损失函数内部完成。
8. **索引取值与独热相乘数学等价** —— 实测同为 `0.574438`，PyTorch 选前者因为省掉一个 57942 长的零向量。

---

## 参考来源

- CrossEntropyLoss — PyTorch 官方文档：https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html —— 输入/目标形状、`(N,C,d1,...,dK)` 的 per-pixel 用法、"unnormalized logits" 措辞均出自这里
- Tensor View Operations on Non-contiguous tensors — PyTorch 论坛：https://discuss.pytorch.org/t/tensor-view-operations-on-non-contiguous-tensors/136316 —— `view` 在非连续张量上的报错原文与 stride 解释
- What's the Difference Between Reshape and View in PyTorch — GeeksforGeeks：https://www.geeksforgeeks.org/deep-learning/whats-the-difference-between-reshape-and-view-in-pytorch —— `view` 要求连续、`reshape` 自动复制
- PyTorch Tensor Implementation Details — apxml：https://apxml.com/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/tensor-implementation —— `.contiguous()` 与 `reshape()` 的选择
- How is teacher-forcing implemented for the Transformer training? — Stack Overflow：https://stackoverflow.com/questions/57099613/how-is-teacher-forcing-implemented-for-the-transformer-training —— `tar_inp = tar[:, :-1]` / `tar_real = tar[:, 1:]` 的通用写法与因果掩码并行的原因
- A Complete Guide to PyTorch Loss Functions — Lightly：https://www.lightly.ai/blog/pytorch-loss-functions —— `(N,C)` 配 `(N,)` 与 `CrossEntropyLoss` = softmax + NLL 的组合关系
