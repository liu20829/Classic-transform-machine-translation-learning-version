# Transformer 解码器理解路径（RNN 对照版）

> 以「从 RNN 过来的人会怎么想」为主线，把解码器的输入输出形式走一遍：训练为什么不用循环、推理为什么还得循环、编码器输出怎么复用
> 关键词：RNN 对照, shifted right, 批量并行, 自回归循环, teacher forcing, KV 复用
> 生成日期：2026-09-13
> 配套文档：`Transformer解码器的输入输出形式-知识概念笔记.md`（同主题的概念查证版，本文件的概念章节与其一致）

---

## 起点：我原来是怎么想的

理解这条路的起点是先把自己的想法写下来。下面这段是原始自述，**它大部分是对的**，只有三处要修——整篇文档就是围绕这三处展开的：

> 解码器的输入是 `target[:,:-1]`，作为批量输入，然后使用 `target[:,1:]` 作为解码器的输出，因为解码器的输出是输入多少就返回多少，比如输入了 51 个词就返回 51 个词，之所以把 `target[:,:-1]` 和 `target[:,1:]` 错开是为了构成预测对，`target[:,:-1]` 的第 0 位作为预测的条件用来预测 `target[:,1:]` 的第 0 位，正好构成 SOS→I 的预测链，之所以和 RNN 不同是进行的批输入和批输出作为快速训练。但是在实际的使用中我们输入中文的「我爱你」，此时解码器的输入起始永远是 SOS，解码器根据编码器的输出作为条件去预测 SOS 的下一个词，得到 I，然后循环解码器输入 `[SOS, I]`，取预测 you，因为是输入多少行就输出多少行所以实际的输出是 I, you，取 -1 位进行流式返回。
>
> 我在捋思路的时候发现每次循环是不是都要重新输入一下「我爱你」让编码器再算一遍，然后拿上一步解码器的输出作为这次解码器的输入？还是整个循环链中编码器只算一次、后续复用它的输出？

**三处要修的：**

| # | 自述原文 | 问题 | 修正在 |
|---|---------|------|--------|
| 1 | 「使用 `target[:,1:]` 作为解码器的输出」 | 它是**标签**，不是解码器的输出。解码器的输出是 `output` | 第四部分 |
| 2 | 「循环输入 `[SOS, I]`，取预测 you」 | 差一位。`[SOS, I]` 后面跟的是 **love**，不是 you | 第五部分 |
| 3 | ——（末尾的疑问） | 编码器**只算一次**，全程复用 | 第五部分 §5.3 |

其余部分（批量输入、错开构成预测对、行数=输入长度、取 -1 位）都对，整篇文档的主线就是把这些「对的」串起来，再把「错的」摆正。

---

## 第一部分 RNN 解码器：循环是被架构逼出来的

### 1.1 RNN 长什么样

RNN 解码器的核心是一行代码——**把上一步的输出喂回输入**：

```python
h = init_hidden()
x = SOS
for t in range(max_len):
    h, y_t = rnn_cell(x, h)   # 输入 x 和上一步的 h
    x = y_t                    # 把这一步的输出作为下一步的输入
```

### 1.2 为什么必须等

`h` 是**算出来的**。位置 1 要动，必须先拿到位置 0 算完的 `h`；位置 2 要动，必须先拿到位置 1 的 `h`。

```
位置0: 需要 h₀ → 算 → 得到 h₁
位置1: 需要 h₁ → 等位置0 算完 → 算 → 得到 h₂
位置2: 需要 h₂ → 等位置1 算完 → 算 → 得到 h₃
```

**这条依赖链没法拆。** 序列长 50，就得老老实实循环 50 次，训练一个 batch 也要循环 50 次。GPU 擅长的是大规模矩阵并行，RNN 这种一步等一步的结构正好用不上。

### 1.3 一句话抓住 RNN 的本质

> **RNN 的「前缀」是算出来的。**

位置 2 想知道前面有什么，答案不在它的输入里——它只知道当前一个词，前面所有的上下文都压缩在那个递推来的隐藏状态 `h` 里。而 `h` 是一步一步算出来的，所以必须等。

这句话是理解整篇文档的钥匙。后文反复出现的困惑，根子都在「把 RNN 的这条性质默认套到了 Transformer 上」。

---

## 第二部分 Transformer 拆掉循环的那一刀

### 2.1 训练代码对比

同样是把 `I love you` 喂进去，两个架构的写法完全不同：

```python
# RNN 解码器 —— 把"看前缀预测下一个"写成显式循环
h = init_hidden()
x = SOS
for t in range(max_len):
    h, y_t = rnn_cell(x, h)    # 一步一个词
    x = y_t
# 循环 50 次，出 50 个词

# Transformer 解码器 —— 没有循环
output = model(input, target[:,:-1])   # 一次，出 51 行
```

### 2.2 前缀：算出来的 vs 现成的

Transformer 凭什么能一次算完？因为它**不需要算前缀**——前缀就在输入张量里摆着：

```python
target[:,:-1] = [SOS, I, love, you, EOS, PAD, ...]
```

位置 2 想知道前面有什么？左边两格就是 SOS 和 I，**它早在张量里躺着了**，不用等谁算出来。

| | RNN | Transformer |
|---|---|---|
| 位置 $t$ 怎么知道「前面有什么」 | 靠上一步传来的隐藏状态 $h$ | 靠**输入张量里现成的词** |
| 所以「前缀」是 | **算出来的**——必须等上一步 | **现成的**——喂进去就有了 |
| 能不能并行 | 不能 | 能 |

### 2.3 掩码：把「只许看左边」一次性盖上矩阵

前缀是现成的了，但有个新问题：**输入张量里连右边的词也一并给了**，位置 2 一伸手就能摸到位置 3 的 you。这在训练时是作弊——答案提前泄露了。

RNN 靠「还没轮到」天然挡住了右边；Transformer 没有这个天然屏障，得自己造一个。造法就是**因果掩码**：

```
允许看到的:          实际盖上去的:
位置0: [SOS]        1 0 0 0 0
位置1: [SOS I]      1 1 0 0 0
位置2: [SOS I love] 1 1 1 0 0
位置3: [SOS I love you]  1 1 1 1 0
```

落到代码上是一行矩阵乘法加一次遮蔽：

```python
# model/attention.py —— 一次矩阵乘法算出全部 51×51 得分，没有位置循环
temp = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale   # q,k: [N,H,51,32]
temp = temp.masked_fill(mask == 0, -1e10)               # 上三角置成极大负数
softmax_out = torch.softmax(temp, dim=-1)               # softmax 后这些位置权重恰好为 0
```

`torch.matmul` 一次算完 51×51，**没有 `for` 循环遍历位置**。全域唯一的循环是层：

```python
# model/decoder.py
for layer in self.layers:            # 7 层，层与层串行
    target, attention = layer(...)    # 每层内部的 51 个位置一次算完
```

### 2.4 结果一样，过程不一样

并行算出来的 51 行，和串行算出来的结果**完全一致**。因为每个位置的「信息视野」不同：

```
位置0 的视野: [SOS]                → 等价于"看 SOS 猜下一个"
位置1 的视野: [SOS, I]             → 等价于"看 SOS,I 猜下一个"
位置2 的视野: [SOS, I, love]       → 等价于"看 SOS,I,love 猜下一个"
```

五个「以为」同时算完。

> **结果等价 ≠ 计算过程如此。** RNN 的「看一个吐一个」是**计算方式**，Transformer 的「看一个吐一个」只是**视野受限的结果**——它压根没有「一个一个」这个动作。

> **白话：** 好比算 `[[1,2],[3,4]] @ [[5,6],[7,8]]`，结果里第 2 行第 2 列那个数确实只用到第 2 行和第 2 列的数。但你不会说「这个矩阵乘法内部先算完第 1 行再算第 2 行」——它是一次算完的。因果掩码同理。

---

## 第三部分 一次喂进去的是什么（训练）

### 3.1 两张切片

数据集里的目标句是这样造的（`tokenizer.py:16`）：

```python
token = [dict.SOS_TOKEN]
token += [dictionary.word2index[word] for word in sentence.split(' ')]
token.append(dict.EOS_TOKEN)
token += [dict.PAD_TOKEN] * (MAX_LENGTH - len(split_sentence))
```

即 `[SOS] + 词 + [EOS] + [PAD]*补足`，总长恒为 $L$（本仓库 $L = 50 + 2 = 52$）。以 `I love you` 为例：

```
下标:     0     1     2      3     4      5  ...  50    51
target:  SOS    I   love   you   EOS    PAD  ...  PAD   PAD
```

从这一个张量切出两张：

```
输入     [─────────── 下标 0 ~ 50，共 51 个 ───────────]
标签        [────────── 下标 1 ~ 51，共 51 个 ──────────]
```

### 3.2 配对规则只有一条

> **输入位置 $t$ 要预测的那个词，就放在标签位置 $t$ 上。**

换成下标说更死板：输入位置 $t$ 装的是 `target[t]`，标签位置 $t$ 装的是 `target[t+1]`，两者差一位——**这就是「错开」的来源**。

措辞上注意**做预测的是输入位置，标签只是答案**。写「标签位置要预测的词」是把主语搞反了。

| 输入位置 $t$ | 装什么 | 它要预测的词 | 落在标签位置 |
|---|---|---|---|
| 0 | SOS | I | 标签位置 **0** |
| 1 | I | love | 标签位置 **1** |
| 2 | love | you | 标签位置 **2** |
| 3 | you | EOS | 标签位置 **3** |

### 3.3 为什么两端各砍一个

配对的道理定了，剩下的是「砍哪两头」，由两条独立的理由推出来：

**先定标签——SOS 不可预测。** 它是人为插进去的起点，不是句子内容，任何句子开头都是它，不存在「预测 SOS」这个任务。所以标签从第 1 个词开始：

```
target         : SOS  I  love  you  EOS  PAD ...
target[:, 1:]  :      I  love  you  EOS  PAD ...     ← 长度 L-1
```

**再定输入——输入的最后一个位置没有对应的标签。** 标签占位置 $0 \ldots L-2$，按配对规则输入位置范围相同，所以输入内容只到 `target[L-2]` 为止：

```
target          : SOS  I  love  you  EOS  PAD ...
target[:, :-1]  : SOS  I  love  you  EOS  PAD ...    ← 长度 L-1
```

| 切片 | 少了哪一头 | 为什么 |
|------|-----------|--------|
| 输入 `target[:,:-1]` | 末尾下标 51 | 它后面已经没有标签了，留着它输入就比标签多一位，预测数和标签数对不上 |
| 标签 `target[:,1:]` | 开头下标 0 | 那是 SOS，不是句子内容，没有对应的预测任务 |

**如果两边都从同一头砍会怎样：**

| 做法 | 结果 |
|------|------|
| 都砍开头（都取 `[1:]`） | 输入变成 `[I, love, you, EOS, ...]`，起点不再是 SOS，位置 0 成了「看到 I 猜 love」——SOS 白加了 |
| 都砍末尾（都取 `[:-1]`） | 标签里含 SOS，等于要求模型预测「第 0 个词是 SOS」；且 EOS 永远拿不到监督信号，模型学不会什么时候该停 |

### 3.4 两张切片走两条路

代码里这四行挨在一起（`train.py`）：

```python
# 题目：整个切片一次性进模型
output, _ = model(input, target[:,:-1])   # 输出 [B, 51, V_tgt]

# 答卷：模型的产出摊平
output = output.contiguous().view(-1, output.shape[-1])   # [B*51, V_tgt]

# 标准答案：切片不进模型，只在这里被赋值
target = target[:,1:].contiguous().view(-1)   # [B*51]

# 批改：答卷与标准答案在这一行碰头
loss = criterion(output, target)
```

**`target[:, 1:]` 从头到尾没出现在 `model(...)` 的实参里。** 两张切片真正碰头的位置是最后那行损失函数，不是模型入口。

> **白话：** 想象成考试——题目（`target[:,:-1]`）发给学生，等学生交卷（`output`），老师拿标准答案（`target[:,1:]`）批改。**标准答案不给学生看**，给了就是作弊。

一个读代码的陷阱：`train.py:100` 是赋值语句，把变量名 `target` 覆盖了。覆盖前是 `[B, 52]` 的原串，覆盖后是 `[B*51]` 的标签——名字复用了，前后不是同一个东西。

---

## 第四部分 一次吐出来的是什么

### 4.1 51 行，每行 $V_{tgt}$ 个分数

`output` 的形状是 `[B, 51, V_tgt]`。取单个样本 `output[0]` 得 `[51, V_tgt]`：

| 维度 | 大小 | 含义 |
|------|------|------|
| 第 1 维 | 51 | 位置数——每个位置一行 |
| 第 2 维 | `V_tgt` | 词表大小——每行对词表里**每一个词**各打一个分 |

**每行装的不是一个词，是一整行分数。** 用一个 6 词的小词表走一遍（`index: 0=PAD, 1=SOS, 2=EOS, 3=I, 4=love, 5=you`）：

| 第几行 | 该行内容（对 6 个词的打分） | `argmax` 取到 |
|---|---|---|
| 0 | `[-2.1, -3.4, -1.8, **8.7**, 2.3, -0.5]` | index 3 → **I** |
| 1 | `[-2.5, -3.1, -0.9, 1.2, **9.1**, 0.3]` | index 4 → **love** |
| 2 | `[-1.9, -3.3, 1.1, 0.4, 2.7, **8.9**]` | index 5 → **you** |
| 3 | `[-1.5, -3.0, **9.4**, -0.2, 1.1, 0.7]` | index 2 → **EOS** |

打完所有候选的分，`argmax` 挑出最高分的下标，**那个下标本身就是词 id**。

> **注意计数基准不同：** 行号从 0 数、「第几个词」从 1 数，两者指同一个位置，不是错位。

为什么必须是一整行而不是直接给一个词：**模型事先不知道要输出哪个词**，只能对每个候选都打个分，再由 `argmax`（推理）或交叉熵损失（训练）去挑。

### 4.2 输出层只有一个 `Linear`，不按位置分头

这里有个容易想歪的地方：**并没有「51 个输出头」**。

```python
# model/decoder.py:94
self.linear = nn.Linear(dim_model, tgt_vocab_size)    # Linear(256, V_tgt)
...
# model/decoder.py:104
output = self.linear(target)
```

整个输出层就这一个 `Linear`。为什么输入 51 个位置就出 51 行？因为 `nn.Linear` 作用在**最后一维**上，对前面的维度是**广播**的：

```
[256]          → Linear → [V_tgt]           单个位置：做一次 256→V 的映射
[B, 51, 256]   → Linear → [B, 51, V_tgt]    51 个位置：同一个 Linear 做了 51 次
[B, 1,  256]   → Linear → [B, 1,  V_tgt]    1 个位置：做一次，出 1 行
```

**是同一个变换被施加了 51 次，不是 51 个不同的变换。** 位置维对 `Linear` 来说就像批量维——来多少算多少。

训练和推理**共用同一份权重**（`translate.py:51` 加载的就是训练产出的 `best_model.pt`），推理时输入 1 个位置就自动出 1 行，一个字都不用改。

### 4.3 训练时 51 行全用

| | 假设只产出末尾一行 | 实际 |
|---|---|---|
| `output` 形状 | `[B, V_tgt]` | `[B, 51, V_tgt]` |
| 展平后 | `[B, V_tgt]` | `[B*51, V_tgt]` |
| 预测个数 | B 个 | B*51 个 |
| 标签个数（`target[:,1:]` 展平） | B*51 个 | B*51 个 |
| 能否计算 loss | **形状不匹配，直接报错** | 一一对应 |

**标签有多少个，预测就必须有多少个。** 所以训练时 51 行**全部参与**损失计算，各自跟一个标签比对。

---

## 第五部分 推理：循环又回来了

### 5.1 为什么绕不开循环

训练时那个漂亮的并行，在推理时**用不了**。因为并行有个前提：前缀得是现成的。

而推理时你手上只有一句中文，**英文前缀一个字都还没有**——它就是你要生成的东西。没有现成的前缀，就只能吐一个、拼回去、再喂进去。

| | RNN 的循环 | Transformer 推理的循环 |
|---|---|---|
| 为什么绕不开 | **架构决定的**——隐藏状态 $h$ 必须从上一步传过来 | **任务决定的**——没有答案可喂，只能自己长 |
| 如果给它完整答案 | 还是得循环，架构就这样 | 就不循环了——**那就是训练** |

关键在这里：

> **Transformer 不是「不需要循环的架构」，而是「有答案时可以不循环的架构」。**

所以它的卖点要说得准：**不是「永远不循环」，是「训练时可以并行」**。训练才是瓶颈——几十个 epoch、几百万个句子，RNN 每句都要循环 50 次，Transformer 一次前向就完事。推理的循环则是自回归生成这个任务的固有性质，跟架构无关：RNN、Transformer、任何自回归模型，推理时都得一个一个吐。

### 5.2 时间线

本仓库 `translate.py:70-87`：

```python
# translate.py:70
target_tokens = [dict.SOS_TOKEN]

# translate.py:73
for i in range(max_len):
    target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
    target_mask = model.make_target_mask(target_tensor)
    output, attention = model.decoder(target_tensor, encoded_input, target_mask, input_mask)
    # translate.py:82  只取最后一个位置的预测
    pred_token = output.argmax(2)[:,-1].item()
    # translate.py:84  拼回输入
    target_tokens.append(pred_token)
    # translate.py:87  遇到 EOS 停止
    if pred_token == dict.EOS_TOKEN:
        break
```

逐轮展开：

```
第1次  输入 [SOS]                  → 输出 1 行 → 取行0 → I
第2次  输入 [SOS, I]               → 输出 2 行 → 取行1 → love
第3次  输入 [SOS, I, love]         → 输出 3 行 → 取行2 → you
第4次  输入 [SOS, I, love, you]    → 输出 4 行 → 取行3 → EOS
                                                      ↓
                                                    停止
累计答案:  I  →  I love  →  I love you  →  I love you EOS
```

**输出行数 = 输入长度**，每轮都取 `[:,-1]`——这两条你原来的理解都对。

三处细节：

| 细节 | 代码依据 | 原因 |
|------|---------|------|
| 循环从 `[SOS]` 开始，长度每轮 +1 | `:70` 初始化，`:84` 追加 | 没有答案可喂，只能自己长出来 |
| 只取最后一个位置的预测 | `:82` 的 `[:,-1]` | 前面位置是在复现已知的词，只有最后一个才是「新词」 |
| 掩码每轮重新构造，尺寸逐轮变大 | `:76` 在循环内 | 第 $i$ 轮输入长度是 $i+1$，掩码为 $(i+1) \times (i+1)$ |

**一个容易被忽略的推论：** 既然前面那些行被丢掉，那它们预测出什么都不影响结果。位置 0 在第 2 轮里的输出，未必等于输入里的 `I`——但**位置 1 的输入是输入张量里的 `I`，不是行 0 的输出**，所以生成结果不受影响。

**最后一步的两个小细节：** EOS 是**先追加再判断**（`:84` 在 `:87` 之前），所以 `target_tokens` 最终含 EOS；返回时 `:93` 的 `[1:-1]` 把开头的 SOS 和结尾的 EOS 一起去掉，才得到 `"I love you"`。

### 5.3 编码器只跑一次，全程复用

这是自述末尾那个疑问的答案。**看循环的边界就明白了**：

```python
# ── 循环外：编码器只跑这一次 ──
input_mask = model.make_input_mask(input_tensor)
with torch.no_grad():
    encoded_input = model.encoder(input_tensor, input_mask)   # ← 整个推理只算一次

target_tokens = [dict.SOS_TOKEN]

for i in range(max_len):                                       # ← 循环从这里开始
    target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
    target_mask = model.make_target_mask(target_tensor)
    with torch.no_grad():
        output, attention = model.decoder(target_tensor, encoded_input, target_mask, input_mask)
        #                                              ^^^^^^^^^^^^^ 每轮直接拿现成的，不重算
```

| | 跑几次 | 为什么 |
|---|---|---|
| **编码器** | **1 次** | 源句「我爱你」全程一个字没变，输出自然不会变 |
| **解码器** | 最多 50 次 | 每轮前缀都长了一个词，输入变了就得重算 |

**如果每轮把「我爱你」重喂一遍会怎样？** 结果完全一样，但白白浪费 50 倍算力。**源句不依赖已生成的词**，这是能复用的根本原因。

**顺带一个代价：** 解码器每轮都要把整条前缀重算一遍（输出行数 1、2、3、4……累起来约 $n^2/2$），而 RNN 每次只算一个新 cell、总量 $O(n)$。所以不打 KV cache 优化的话，**Transformer 推理并不比 RNN 快**，甚至更慢。它的优势全在训练侧的并行。

### 5.4 流式输出：取 -1 位 ≠ 立刻发给用户

和 AI 对话时看到的打字机效果，做法是每轮取 `[:,-1]` 就吐出去。**取哪一行这件事，流式和非流式完全一样**，差别只在什么时候把结果交出去：

| | 本仓库 `translate.py` | 流式（ChatGPT 那种） |
|---|---|---|
| 每轮取哪一行 | `[:,-1]` | `[:,-1]` —— **完全一样** |
| 给用户的时机 | 循环全部结束后，一次性 | 每轮立刻吐一个 |
| 实现方式 | `return` | `yield`（生成器） |
| 用户观感 | 整句「啪」地出现 | 逐词往外冒 |

本仓库是 `return`（`:93`），循环跑完才返回整句，所以**它不是流式的**。流式的写法只是把 `return` 换成 `yield`：

```python
for i in range(max_len):
    ...
    yield pred_token     # ← 每轮立刻吐出去，调用方边收边显示
```

**一个容易混的点：** 流式给人的感觉像「模型一次算一个词」，但实际不是——**模型每轮算的仍然是一整行分数向量**（长度 `V_tgt`），只是被 `argmax` 成一个词立刻发出去。**计算粒度没变，只是传输粒度变小了。**

---

## 第六部分 训练 vs 推理总对照

| 维度 | 训练 | 推理 |
|------|------|------|
| 代码位置 | `train.py:96` | `translate.py:73-87` |
| 目标侧输入来源 | 真实答案的错位切片 `target[:,:-1]` | 模型自己已生成的前缀 |
| 输入长度 | 固定 51，一次喂完 | 从 1 逐步增长到 EOS 或 `max_len` |
| 前向次数（解码器） | **1 次**，所有位置同时算 | **`max_len` 次**（本仓库 50） |
| 前向次数（编码器） | 1 次 | **1 次**（循环外，复用） |
| 「只看得到前面」如何实现 | 因果掩码**人造**的假象 | 未来位置**客观上不存在** |
| 每轮输出的用途 | 全部 51 行都参与损失计算 | 只取最后一行的输出 |
| 参数更新 | 有反向传播 | 无，`with torch.no_grad()` |
| 给结果的时机 | —— | 本仓库一次性 `return`；流式用 `yield` |
| 有没有循环 | **没有**（这是 Transformer 的核心优势） | **有**（和 RNN 在接口上没区别） |

**一句话概括整篇：** 训练时是「假装」自回归（掩码造的），推理时是「真的」自回归（循环造的）。shifted right 存在的全部理由，是让训练能够并行——若无此技巧，训练也得像 RNN 那样一步步循环。

---

## 第七部分 我的自述逐条订正

| 自述内容 | 判定 | 说明 |
|---|---|---|
| 解码器输入是 `target[:,:-1]`，批量输入 | ✓ | |
| 输出是输入多少返回多少，51 个词返回 51 行 | ✓ | 由 `Linear` 广播决定，见 §4.2 |
| 错开是为了构成预测对 | ✓ | 配对规则见 §3.2 |
| `[:-1]` 第 0 位预测 `[1:]` 第 0 位，构成 SOS→I | ✓ | |
| 和 RNN 不同在于批输入批输出、训练快 | ✓ | 差别在「前缀是算出来的 vs 现成的」，见 §2.2 |
| 实际使用中起始输入永远是 SOS | ✓ | |
| 预测 SOS 的下一个词，得到 I | ✓ | |
| 取 `-1` 位 | ✓ | 每轮都取 `[:,-1]`，见 §5.2 |
| **用 `target[:,1:]` 作为解码器的输出** | ✗ | 它是**标签**，模型看不见。解码器的输出是 `output`。见 §3.4 |
| **循环输入 `[SOS, I]` 取预测 you** | ✗ | 应该是 **love**。差一位——输入里最后是 I，它后面跟的是 love。见 §5.2 |
| **所以实际输出是 `[I, you]`** | ✗ | 第 2 轮输出 2 行，`argmax` 是 `[I, love]`。见 §5.2 |
| 编码器是否每轮重跑？ | 答：**不重跑，只算一次** | 见 §5.3 |

---

## 附录 概念查证区

> 以下内容与配套文档 `Transformer解码器的输入输出形式-知识概念笔记.md` 一致，是主线之外的「查证材料」：标准命名、论文出处、工业界对照、学术争议、仓库调用链细节。主线读懂了再看这里。

### A.1 shifted right —— 标准命名与论文出处

该做法在原始论文 *Attention Is All You Need*（Vaswani et al., 2017）图 1 的右下角被标注为 **"Outputs (shifted right)"**。论文正文对应段落原文：

> "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$."

译：我们同样修改了解码器堆叠中的自注意力子层，以阻止某个位置关注它之后的位置。这种掩码，结合输出嵌入被偏移一位这一事实，共同保证了位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

这段话同时点出了**两个机制**，不可混淆（见 A.2）。

### A.2 错位与因果掩码的分工

**shifted right 和 causal mask 是两个独立机制，解决两个不同问题，互换不可。**

| 机制 | 解决的问题 | 若不做的后果 |
|------|-----------|-------------|
| shifted right（错位一位） | 防止模型看到「当前要预测的那个词」本身 | 退化为 copy 映射 |
| causal mask（因果掩码） | 防止模型看到「未来位置的词」 | 位置 $i$ 能看到 $i+1 \ldots L-1$，训练时等于把整句答案提前泄露 |

两者的失效方向不同，因此**必须同时存在**：

- 只做 mask 不做 shift：位置 $i$ 看不到未来，但看得到 `target[i]` 自己——仍然是 copy 问题
- 只做 shift 不做 mask：位置 $i$ 看不到 `target[i]` 自己，但看得到 `target[i+2]` 等未来词——等于作弊

本仓库里两个机制各由一处代码实现：

```python
# 机制一：错位 —— train.py
output, _ = model(input, target[:,:-1])   # 去尾切片，让目标侧看不到"当前要预测的那个词"

# 机制二：因果掩码 —— model/transformer.py 的 make_target_mask
def make_target_mask(self, target):
    target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
    # 下三角，遮掉未来位置
    target_tril_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]), device=self.device)).bool()

    target_mask = target_pad_mask & target_tril_mask   # 两个遮蔽条件按位与
    return target_mask
```

注意 `torch.tril` 的方阵尺寸取自 `target.shape[1]`，即解码器输入的实际长度 51，因此掩码为 $51 \times 51$。错位切完后，下游每一步的尺寸自动自洽，不再需要原串的 52。

**「两个机制不能互换」怎么实测** —— 本仓库没有覆盖这条的单测，要验证需跑训练并做对照。两个实验：

| 实验 | 改动 | 预期现象 |
|------|------|---------|
| 取消 shift | `train.py:96` 改为 `model(input, target)`，同时 `train.py:100` 改为 `target.contiguous().view(-1)`（两边都恢复成 52 长才能对齐形状） | 位置 $i$ 的输入含 `target[i]`，标签也是 `target[i]`，模型学恒等映射即可，损失迅速塌到接近 0 |
| 取消 mask | `model/transformer.py:36` 的 `target_pad_mask & target_tril_mask` 改为只用 `target_pad_mask` | 位置 $i$ 能看到未来词，损失异常低 |

两个实验的共同特征是**训练损失很好看，但拿去 `translate.py` 推理会崩**——这正是区分「真的学会了」和「偷偷抄了」的判据。注意第一个实验必须先对齐长度，否则会先在形状检查处直接报错。

### A.3 工业界对照

Hugging Face `transformers` 库提供函数 `shift_tokens_right`，做的是同一件事：把标签整体右移一位，并把首位填入 `decoder_start_token_id`（即 SOS）。

| 抽象概念 | 本仓库（手写） | HuggingFace（库函数） |
|---------|--------------|---------------------|
| 起始标记 | `dict.SOS_TOKEN` | `decoder_start_token_id` |
| 错位操作 | `target[:,:-1]` 与 `target[:,1:]` 成对切片 | `shift_tokens_right(labels, ...)` |
| 因果掩码 | `make_target_mask` 中的 `torch.tril` | `is_decoder=True` 时自动生成 causal mask |

命名上本文统一使用论文的 **shifted right**（错位右移），不使用其他自造说法。

### A.4 为什么编码器的输出要喂给解码器

**交叉注意力的 Q/K/V 归属** —— 解码器层内有两个多头注意力，这是整个编码器-解码器结构的连接点：

| 注意力 | 代码位置 | Q 来源 | K、V 来源 | mask 来源 |
|--------|---------|-------|----------|----------|
| 自注意力 | `model/decoder.py:47` | 目标侧输入 | 目标侧输入 | `target_mask`（遮未来 + 遮 PAD） |
| 交叉注意力 | `model/decoder.py:51` | 目标侧输入 | **编码器输出** | `input_mask`（只遮 PAD） |

```python
# model/decoder.py:47
attention_1, _ = self.attention1(target, target, target, target_mask)
# model/decoder.py:51
attention_2, attention = self.attention2(attention_1_norm, encoded_input, encoded_input, input_mask)
```

注意第 51 行的三个位置参数：第一个是 `attention_1_norm`（目标侧的中间表示），后两个都是 `encoded_input`。这使得：

- 源语言信息**只从 K 和 V 进入**——即源句只被「查询」，不被「改写」
- 目标侧信息从 Q 进入——即「用目标侧的当前状态去问源句：我现在要写第 $i$ 个词，源句里哪部分相关」

**形状明细**（本仓库 `d_model=256`、`n_heads=8`，故 $d_k = 32$）：

| 张量 | 形状 | 来源 |
|------|------|------|
| Q | `[B, 8, 51, 32]` | 目标侧投影 |
| K | `[B, 8, 52, 32]` | 编码器输出投影 |
| V | `[B, 8, 52, 32]` | 编码器输出投影 |
| 得分矩阵 $QK^\top/\sqrt{d_k}$ | `[B, 8, 51, 52]` | — |
| 加权结果 | `[B, 8, 51, 32]` | 得分矩阵 @ V |

得分矩阵是 $51 \times 52$ 的**非方阵**——行是目标侧位置（51 个），列是源句位置（52 个）。**非方阵因此不可能有下三角掩码**，这也再次说明为什么 `input_mask` 只做 padding 遮蔽。

**为什么必须有这一路** —— 没有交叉注意力，解码器退化为一个纯语言模型，只能估计 $P(\text{目标句})$，而不知道要翻译的是什么。加上后估计的是条件分布：

$$P(y_1, y_2, \ldots, y_m \mid x_1, \ldots, x_n) = \prod_{t=1}^{m} P(y_t \mid y_1, \ldots, y_{t-1}, \; \bar{x}_1, \ldots, \bar{x}_n)$$

其中 $x$ 为源句词 id，$\bar{x}$ 为编码器输出的上下文向量，$y_t$ 为目标词。这正是神经机器翻译的训练目标：**最大化 $P(\text{target} \mid \text{source})$**。

注意右侧连乘的每一项条件里都含 $\bar{x}_{1..n}$ 全部——因为交叉注意力让每个解码位置都能看到整个源句。这解决了 RNN 编码器-解码器架构中源句信息必须被压进单个定长向量的瓶颈问题。

**用具体数字说** —— 假设目标句是 `I love you`，长度 3，则

$$P(\text{I love you} \mid \text{源句}) = P(\text{I} \mid \text{SOS}, \text{源句}) \times P(\text{love} \mid \text{SOS I}, \text{源句}) \times P(\text{you} \mid \text{SOS I love}, \text{源句})$$

三项分别由位置 0、1、2 的输出给出。

### A.5 两个 mask 的差异

```python
# model/transformer.py —— 两个掩码的构造，并排看差异
def make_input_mask(self, input):
    input_mask = (input != self.padding_index).unsqueeze(1).unsqueeze(2)
    return input_mask                                    # 到此为止，只有 padding 遮蔽

def make_target_mask(self, target):
    target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
    target_tril_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]), device=self.device)).bool()
    target_mask = target_pad_mask & target_tril_mask     # 多了一步：再与上下三角
    return target_mask
```

| | `input_mask` | `target_mask` |
|---|---|---|
| 内容 | `(src != PAD)` | `(tgt != PAD) & tril` |
| 有下三角吗 | 无 | 有 |
| 为什么 | 源句是**已知完整信息**，读它的时候不需要隐藏任何部分——翻译时可以反复看整句原文 | 目标句处于**生成过程中**，位置 $i$ 只能看到已生成的前缀，未来位置必须遮住 |

一句话：**源句是「给定条件」，不遮未来；目标句是「生成过程」，必遮未来。**

**推理时为什么还需要 `make_target_mask`** —— 虽然此刻输入序列本身就是「从 SOS 开始逐步增长」的，不存在未来词泄露问题，但 **padding mask 部分仍然要生效**。这说明两个 mask 机制在整个推理循环中都是持续在场的。

### A.6 代价：exposure bias

**定义**：因为模型在训练阶段只接触真实前缀、在推理阶段只接触自己生成的前缀，两个阶段的输入分布不一致，导致中间过程不一致；推理时一旦某步预测错误，该错误会成为后续步骤的输入，可能累积放大。此现象由 Ranzato et al. (2015) 与 Bengio et al. (2015) 命名。

| 维度 | 说明 |
|------|------|
| 成因 | 训练分布（真实前缀）与推理分布（模型前缀）不匹配 |
| 表现 | 推理时出现重复、不连贯、提前结束 |
| 常见缓解手段 | scheduled sampling（按衰减概率混入模型自身预测）、强化学习目标（REINFORCE）、对抗训练、知识蒸馏 |
| 学术争议 | He et al. (2019) 的 *Exposure Bias versus Self-Recovery* 认为该偏差并不如通常认为的严重，语言模型具备「自我恢复」能力，即使从打乱或随机的前缀出发仍能生成合理文本 |

这是 teacher forcing 这套机制自带的问题，不是本仓库的实现缺陷。

**「输入分布不一致」具体不一致在哪：**

| 场景 | 第 3 步的输入可能是什么 |
|------|---------------------|
| 训练 | `SOS I love`（一定是正确的前三个词） |
| 推理 | `SOS I loves`（如果第 2 步预测错了，这里就是错的词） |

模型在训练中从没处理过 `SOS I loves` 这种输入，而推理中会遇到。

### A.7 本仓库的完整调用链

**数据成形：**

```
原始平行语料 data/cs-en/{cs.txt, en.txt}
        ↓  dictionary.create_dictionary('cs', 'en')
词表 + 句子列表
        ↓  tokenizer.tokenize(sentence, dic, utils.max_sent_len)   [tokenizer.py:16]
        ↓  [SOS] + 词 + [EOS] + [PAD]*(50 - 词数)
定长 52 的整数序列
        ↓  dataloader.get_dataloader(...)
train_dataloader / valid_dataloader
```

**两路输入汇合的地方是 `Transformer.forward`：**

```python
# model/transformer.py —— 两路输入在这里汇合
def forward(self, src, tgt):
    # 1. 跑编码器：源句只遮 PAD，不遮未来
    src_mask = self.make_input_mask(src)
    encoded_input = self.encoder(src, src_mask)

    # 2. 跑解码器：目标侧既要遮 PAD 又要遮未来
    target_mask = self.make_target_mask(tgt)
    output, attention = self.decoder(tgt, encoded_input, target_mask, src_mask)
    return output, attention
```

两个容易看漏的点：

- 形参 `tgt` 就是 `train.py` 传进来的 `target[:,:-1]`，不是原串
- `src_mask` 被原样当作第 4 个实参传进解码器，在解码器的签名里改名叫 `input_mask`。**同一个掩码对象，两处叫法不同**

**损失函数：**

```python
# train.py
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_TOKEN)
```

**模型入口不止训练一处**，验证函数 `evaluate()` 里也有一处，两边的切片写法完全一致：

```python
# 训练路径
output, _ = model(input, target[:,:-1])
output = output.contiguous().view(-1, output.shape[-1])
target = target[:,1:].contiguous().view(-1)
loss = criterion(output, target)

# 验证路径（evaluate 函数内）—— 切片写法一模一样，只是变量名不同
output, _ = model(input, target[:, :-1])
output_reshape = output.contiguous().view(-1, output.shape[-1])
target_reshape = target[:, 1:].contiguous().view(-1)
loss = criterion(output_reshape, target_reshape)
```

两处唯一的实质差别是验证多了 `model.eval()` 与 `torch.no_grad()`。这说明 shifted right 是这套训练范式的一部分，与是否更新梯度无关。

另一个值得注意的差别：验证时的 BLEU 分数是在**同一次前向传播的输出**上逐位置取 argmax 得到的：

```python
# evaluate 函数内，逐句统计 BLEU
for j in range(target.size(dim=0)):
    output_words = output[j].max(dim=1)[1]
```

而不是像 `translate.py` 那样逐词循环生成。前者是在真实前缀的辅助下「读」出来的译文，后者才是模型独立生成的能力。评估时若想反映真实推理表现，以后者为准。

**推理路径一览：**

| 步骤 | 代码位置 | 说明 |
|------|---------|------|
| 造源句的 padding mask | `translate.py:63` | 只遮 PAD，无下三角 |
| **编码器跑一次，在循环外** | `translate.py:67` | 源句不依赖已生成的词，算一次即可 |
| 初始化 `[SOS]` | `translate.py:70` | 自回归的起点 |
| 逐词循环 | `translate.py:73` | 最多 `max_len` 轮 |
| 每轮重造目标掩码 | `translate.py:76` | 尺寸随输入增长 |
| 循环内只调解码器 | `translate.py:79` | 调用的是 `model.decoder` 而非 `model` |
| 取最后一个位置的 argmax | `translate.py:82` | 每轮只取一个新词 |
| 拼回输入序列 | `translate.py:84` | 构成下一轮的输入 |
| 遇 EOS 提前结束 | `translate.py:87` | — |
| 去掉 SOS 与 EOS 后输出 | `translate.py:93` 的 `[1:-1]` | 首尾标记不属于译文 |

### A.8 两处「忽略」的区别

代码里有两处地方在「忽略」某些东西，它们作用的层面不同，不可混为一谈：

| | 前向的 mask | 反向的 ignore_index |
|---|---|---|
| 生效位置 | `model/attention.py` 的 `scaled_dot_product_attn` | `train.py` 的损失计算 |
| 作用层面 | 注意力权重计算阶段 | 损失计算阶段 |
| 手段 | `masked_fill(mask == 0, -1e10)` 使 softmax 后权重为 0 | `CrossEntropyLoss(ignore_index=PAD)` 使该位置不贡献梯度 |
| 忽略谁 | 被遮位置的**注意力权重** | 标签为 PAD 的**位置** |
| 效果 | 模型不会「看到」那些位置 | 模型不会因那些位置的预测被惩罚 |

```python
# 前向：mask 在注意力权重里生效 —— model/attention.py
temp = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale
temp = temp.masked_fill(mask == 0, -1e10)   # 被遮位置置成极大负数
softmax_out = torch.softmax(temp, dim=-1)   # softmax 后这些位置的权重恰好为 0

# 反向：ignore_index 在损失里生效 —— train.py
criterion = nn.CrossEntropyLoss(ignore_index=dict.PAD_TOKEN)
loss = criterion(output, target)            # 标签为 PAD 的位置不贡献梯度
```

### A.9 易混淆点汇总表

| 容易混的点 | 正确表述 |
|-----------|---------|
| 解码器只有一路输入 | 两路：目标侧错位切片（Q 来源）+ 编码器输出（K/V 来源） |
| `target[:,1:]` 也要喂进模型 | 不。它是标签，只在损失函数处使用 |
| 一次 `model(...)` 调用只产出一个预测 | 产出 51 个，每个输入位置各一个 |
| 切片整片进模型 = 每个位置都看得到整片 | 切片确实一次性全给，但**每个输出位置只看得到它自己和它左边的输入位置**（因果掩码） |
| 训练时模型也是「看一个吐一个」 | 训练时 51 个位置一次前向算完，靠掩码「假装」；逐个循环只发生在推理 |
| 训练时位置之间会「算完一个喂给下一个」 | 那是 RNN 的机制。Transformer 的位置间是并行的，位置 1 吃的是输入张量里的 `target[1]`，**不是位置 0 的输出** |
| 「看前缀预测下一个」意味着必须按顺序算 | 「前缀」在 Transformer 里是输入张量里现成的，不是算出来的；顺序感由掩码伪造，非计算所需 |
| Transformer 永不循环 | 训练不循环（并行）；**推理照样循环**（`translate.py:73`），和 RNN 在接口上没区别 |
| Transformer 不循环所以推理更快 | 反了。不打 KV cache 时，推理每次重算整条前缀，总量 $O(n^2)$，比 RNN 的 $O(n)$ 更慢；优势只在训练并行 |
| shifted right 就是因果掩码 | 两个独立机制，一个防 copy，一个防看未来，必须同时存在 |
| 训练时「已经写了的词」是模型写的 | 是直接喂入的真实答案前缀（teacher forcing） |
| `encoded_input` 是词 id | 是浮点上下文向量 `[B, L, d_model]`，词 id 只在编码器入口出现一次 |
| 交叉注意力的 Q 来自源句 | Q 来自目标侧，K/V 来自编码器输出——源句只被「查询」不被「改写」 |
| 源句掩码也该有下三角 | 不该。源句是给定条件，翻译时可反复看整句；只有目标侧才遮未来 |
| 输出层有 51 个「头」，每头管一个位置 | 只有一个 `Linear`，对位置维广播；输入几个位置就出几行 |
| 推理时编码器每轮重跑 | 只跑一次，在循环外；变的是解码器那侧的前缀 |
| 流式输出意味着模型一次算一个词 | 计算粒度没变，每轮仍算一整行分数向量；变的只是传输粒度 |
| `target` 变量前后指同一张量 | `train.py:100` 覆盖了变量名，之后指的是标签而非原串 |

### A.10 总结

1. **解码器是条件自回归序列生成器**，输入两路：目标侧错位切片（`target[:,:-1]`，整数 id）与编码器输出（`encoded_input`，浮点向量）。

2. **`target[:,:-1]` 与 `target[:,1:]` 是一对**：同一句话错开一格，前者框住输入窗口（下标 $0 \ldots L-2$），后者框住答案窗口（下标 $1 \ldots L-1$）。两端各砍一个的理由是长度匹配（模型每个位置产生一个预测）+ 语义合理（PAD 之后不存在词，SOS 无需被预测）。

3. **两张切片走两条路**：切片进模型当题目，标签留在外面当答案，它们在损失函数那一行碰头，不在模型入口碰头。

4. **训练不循环，推理照样循环**：Transformer 不是「不需要循环的架构」，而是「有答案时可以不循环的架构」。训练优势来自并行，推理的循环是自回归任务的固有性质，跟架构无关。

5. **输出层只有一个 `Linear`**：`[B, 51, 256] → [B, 51, V_tgt]` 是靠对位置维广播实现的，不是 51 个独立的头。训练时 51 行全用，推理时只用最后一行。

6. **编码器只跑一次**：源句全程不变，编码器输出在循环外算好后被每轮推理复用；只有解码器因为前缀在变而需要反复重算。

7. **shifted right 与 causal mask 分工明确**：前者防「看到要预测的词自己」（copy 问题），后者防「看到未来词」（作弊问题）。只做其一都会产生训练损失很好但实际没学会的假象。

---

## 参考来源

- Vaswani et al., *Attention Is All You Need*, 2017（Figure 1 的 "Outputs (shifted right)" 与 §3.1 掩码段落）：https://arxiv.org/abs/1706.03762
- Hugging Face Transformers 文档，`shift_tokens_right` 与 `decoder_start_token_id`：https://huggingface.co/docs/transformers
- Zero Math AI, *Output Embedding Shifted Right — Why Shift and Causal Masking Solve Different Problems*：https://zeromathai.com/en/output-embedding-shifted-right-en/
- Ranzato et al., *Sequence Level Training with Recurrent Neural Networks*, 2015（exposure bias 命名出处之一）：https://arxiv.org/abs/1511.06732
- Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, 2015：https://arxiv.org/abs/1506.03099
- He et al., *Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation?*, 2019：https://arxiv.org/abs/1905.10617
- 本仓库源码：`model/transformer.py`、`model/decoder.py`、`model/attention.py`、`train.py`、`translate.py`、`tokenizer.py`、`utils.py`
