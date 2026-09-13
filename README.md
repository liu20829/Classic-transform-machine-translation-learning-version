<div id="top"></div>

# 经典 Transformer 机器翻译学习版

> 上游的 Transformer 从零实现 + 我的手写复现、实测脚本与学习笔记
> English version: [README_EN.md](README_EN.md)

---

## 一、这个仓库是什么

| 项 | 内容 |
|---|---|
| **代码来源** | [shashankag14/Transformer-for-Machine-Translation](https://github.com/shashankag14/Transformer-for-Machine-Translation)，PyTorch 从零实现 Transformer，基于 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) |
| **任务** | 捷克语 → 英语机器翻译（PHP 语料） |
| **我做的** | 手写复现、实测验证、学习文档 |
| **改没改上游** | 逻辑一行没改，只在 `model/attention.py`、`model/encoder.py` 补了中文注释 |

一句话说明这个仓库的定位：

> **上游给了一个「能跑」的实现，我做的是把它「拆开看」。**

仓库名叫「学习班」就是这个意思——不是要刷 BLEU 分数，而是把每一处张量形状、每一次内存移动、每一个「为什么这么写」都落到实处。凡是能实测的，我都不靠推算。

---

## 二、目录导航

| 目录 / 文件 | 内容 | 归属 |
|---|---|---|
| `model/` | Transformer 五个模块：`attention.py`、`encoder.py`、`decoder.py`、`position_encoding.py`、`transformer.py` | 上游 |
| `train.py` / `translate.py` / `plot.py` | 训练、推理、画图入口 | 上游 |
| `tokenizer.py` / `dictionary.py` / `dataloader.py` / `utils.py` / `bleu_metric.py` | 数据处理与评估 | 上游 |
| `scripts/` | 数据下载、环境安装、词表导出 | 上游 |
| `data/cs-en/` | 捷克语—英语平行语料 | 数据 |
| `saved_chkpt/` | 训练产出：`best_model.pt` 与词表 | 产出 |
| `results/` | 训练曲线、损失记录、BLEU | 产出 |
| **`my_/手动复现/`** | 我照自己的理解重写的编码器，含冒烟测试与 mask 验证 | **我** |
| **`my_/测试/`** | 实测脚本：形状追踪、内存行为验证 | **我** |
| **`my_/transformer网络结构图/`** | 结构图 | **我** |
| **`my_doc/`** | 学习文档（含交互式架构图） | **我** |
| `Attention-is-all-you-need/` | 论文原文 PDF | 资料 |

---

## 三、快速开始

### 3.1 环境

```bash
sh scripts/requirements.sh
```

### 3.2 数据

语料已在 `data/cs-en/`（`cs.txt` / `en.txt`）。若缺失，用上游脚本重新下载，注意解压后把 `PHP.cs-en.cs` 和 `PHP.cs-en.en` 放进 `data/`：

```bash
sh scripts/download_data.sh
```

### 3.3 训练

```bash
python train.py
```

常用参数（完整列表见第八章的超参表）：

```bash
python train.py --epoch 150 --batch_size 32 --d_model 256 --n_layers 7 \
                --n_heads 8 --ffn_hidden 1024 --dropout 0.15 \
                --max_sent_len 50 --init_lr 1e-4
```

### 3.4 画训练曲线

训练过程中随时可以跑，曲线存到 `results/`：

```bash
python plot.py
```

### 3.5 用训练好的权重做翻译

```bash
python translate.py
```

结果写入 `results/translation_results.txt`。`translate.py` 加载的是 `saved_chkpt/best_model.pt`。

---

## 四、我的学习路径

按实际学习顺序排列：

| 阶段 | 主题 | 产物 |
|---|---|---|
| ① | **位置编码** —— 从 `sin/cos` 公式到「加法怎么变成旋转」 | 2 篇文档 + 1 个交互式频率探索器 |
| ② | **多头注意力** —— 张量形状怎么变、内存里的数据动没动 | 1 篇文档 + 2 个实测脚本 |
| ③ | **解码器** —— 输入输出形式、为什么训练不循环而推理必须循环 | 2 篇文档 + 交互式架构图 |
| ④ | **手写复现** —— 不看源码，照自己的理解重写编码器 | `my_/手动复现/transformer_encoder.py` |

贯穿全程的一条原则：**能实测的绝不推算。**

---

## 五、手动复现与实测

### 5.1 手动复现（`my_/手动复现/`）

| 文件 | 状态 | 内容 |
|---|---|---|
| `transformer_encoder.py` | ✅ 252 行 | 照自己的理解重写的编码器：`feedforwardlayer`、`MultiHeadAttention`、`transformer_encoder_layer`、`positional`、`embedding_positional`、`transformer_encoder` |
| `transformer_decoder.py` | 🚧 空文件 | 待完成 |

`transformer_encoder.py` 的 `__main__` 里带自检，直接运行即可验证：

```bash
python my_/手动复现/transformer_encoder.py
```

自检做了两件事：

1. **冒烟测试** —— 构造虚拟输入（含 PAD），断言输出形状正确、无 NaN
2. **验证 mask 真的生效** —— 改动被遮蔽位置的词，观察真实位置输出不变（实测 `diff_real = 0.0`）、被遮蔽位置输出改变（`diff_pad = 3.85`）；再把 mask 传 `None`，真实位置输出立刻改变（`diff_nomask = 0.82`）

### 5.2 实测脚本（`my_/测试/`）

| 脚本 | 行数 | 实测什么 |
|---|---|---|
| `shape_trace.py` | 71 | 用项目**真实代码**跑一遍 `scaled_dot_product_attn`，打印每一步张量形状，并检验 PAD 行的 softmax 行为 |
| `view_permute_memory.py` | 40 | 实测 `view` / `permute` 的内存行为：两者都只改 stride 不搬数据（`data_ptr` 不变），只有 `.contiguous()` 会真正复制 |
| `bb.py` | 19 | 位置编码公式实验 |
| `aa.py` | 5 | 草稿 |

运行方式（用你本机的 Python 路径）：

```bash
python my_/测试/shape_trace.py
python my_/测试/view_permute_memory.py
```

`shape_trace.py` 的实测输出中有一处**推翻了直觉**：PAD 作为 query 的行并不会得到均匀分布——真实键上的权重和是 1.0，填充键上是 0.0。这是靠跑出来的，不是推出来的。

---

## 六、学习文档（`my_doc/`）

### 6.1 文档清单

| 文档 | 行数 | 主题 |
|---|---|---|
| `20260907_位置编码-从加法到旋转.md` | 497 | 以二维向量为最小例子，顺序推导位置编码的完整逻辑链 |
| `20260907_算例-加位置编码的变换步骤.md` | 164 | 4 维嵌入下，位置指纹如何由代码逐步生成，以及「加进词向量 → 点积」的完整变换链 |
| `多头注意力的张量形状与内存布局-知识概念笔记.md` | 379 | 形状怎么变 → 内存里的数据动没动 → 为什么必须换 → 那「相似度」到底算什么 |
| `Transformer解码器的输入输出形式-知识概念笔记.md` | 804 | 解码器吃什么、吐什么、为什么这样切；按提问链组织，含论文原文与暴露偏差等理论材料 |
| `Transformer解码器理解路径-RNN对照版.md` | 787 | 同主题的另一种讲法：以「从 RNN 过来的人会怎么想」为主线 |

后两份是**同主题的两个版本**，可互为参考：「知识概念笔记」按问题链组织，「RNN 对照版」按理解路径组织。

### 6.2 交互式产物

| 文件 | 说明 |
|---|---|
| `transformer-架构图/index.html` | 交互式架构图：总览 + 编码器/解码器/注意力三个子图，单击节点跳转右侧代码说明，双击虚线节点展开子图。浏览器直接打开 |
| `pe-frequency-explorer.html` | 位置编码频率探索器：可视化不同维度的波长分布 |

---

## 七、上游项目说明

> 以下为上游作者的原始说明，保留以便溯源。

A PyTorch implementation of Transformers from scratch for Machine Translation on PHP Corpus dataset[1] (Czech->English) based on "Attention Is All You Need" by Ashish Vaswani et. al.[2]. The motive to create this repository is not to implement a state-of-the-art model for Machine Translation, but to get a hands-on experience in implementing the Transformer architecture from scratch.

_**Modification done over the baseline[2] :**_

- _Increased the model depth and improved the validation loss as well as BLEU score with a small margin._
- _Performed ablation study to check the effects of Label Smoothening, vector dimensioanlity, rescaling of word embedding, and varying the size of hidden layer in feed-forward network_

### 7.1 Transformer Architecture ([model/](model/))

<img src="https://user-images.githubusercontent.com/74488693/146267612-aa100838-d75f-48ec-b5d5-ce3755687cb5.png" height="700" width="500">

### 7.2 Multi Headed Attention Block ([attention.py](model/attention.py))

<img src="https://user-images.githubusercontent.com/74488693/144745249-5c99709d-0446-45fc-a4cb-f0428ead371e.png" height="300" width="600">

### 7.3 Computing Attention using Key, Query and Value

<img src="https://user-images.githubusercontent.com/74488693/146843949-2ae064f2-49da-4c99-ac25-690a8b4fd910.png" height="600" width="500">
<img src="https://user-images.githubusercontent.com/74488693/146268694-0c8517a1-5795-4efa-a51b-23bae6fab520.png" height="90" width="350">

### 7.4 Positional Encoding using sin/cos ([position_encoding.py](model/position_encoding.py))

<img src="https://user-images.githubusercontent.com/74488693/146268889-723d15a5-2d18-48ba-85a9-936f72ce646f.png" height="90" width="340">

### 7.5 数据集

PHP Corpus Czech-English

| 项 | 值 |
|---|---|
| 语料总句数 | ~33,000 |
| 词数 < 50 的去重句 | 5464 |
| 训练集 | 4371 |
| 验证集 | 874 |
| 测试集 | 219 |
| 源语言（捷克语）词表 | 8891 |
| 目标语言（英语）词表 | 4564 |

### 7.6 超参数（`train.py`）

| 参数 | 说明 | 本项目取值 | 论文取值 |
| --- | --- | --- | --- |
| `--epoch` | 训练轮数 | 150 | N/A |
| `--batch_size` | 批大小 | 32 | N/A |
| `--d_model` | 词嵌入维度 | 256 | 512 |
| `--n_layers` | 编码/解码层数 | 7 | 6 |
| `--n_heads` | 注意力头数 | 8 | 8 |
| `--ffn_hidden` | 前馈层隐藏单元数 | 1024 | 2048 |
| `--dropout` | Dropout 概率 | 0.15 | 0.1 |
| `--max_sent_len` | 句子最大长度 | 50 | N/A |
| `--init_lr` | 初始学习率 | 1e-4 | N/A |
| `--scheduler_factor` | 学习率衰减系数 | 0.9 | 0.9 |
| `--optim_adam_eps` | Adam epsilon | 5e-9 | 1e-9 |
| `--optim_patience` | 学习率衰减前的等待轮数 | 8 | N/A |
| `--optim_warmup` | 优化器 warmup | 16000 | 4000 |
| `--optim_weight_decay` | 权重衰减 | 5e-4 | N/A |
| `--clip` | 梯度裁剪阈值 | 1.0 | N/A |
| `--seed` | 随机种子 | 1111 | N/A |
| `--label_smooth_eps` | 标签平滑系数 | 0.1 | 0.1 |
| `--early_stop_patience` | 早停耐心值 | 20 | N/A |

### 7.7 正则化手段

按论文[2]采用两种**仅在训练阶段生效**的正则化：

1. **残差 Dropout（0.15）** —— 加在嵌入（词向量 + 位置编码）以及编码器/解码器每个子层的输出上。用 `nn.Dropout()` 而非 `nn.functional.dropout`，以便 `eval()` 时自动关闭。
2. **标签平滑（eps=0.1）** —— one-hot 标签会鼓励过大的 logit 差距，使模型过于自信而容易过拟合；标签平滑通过压小 logit 差距缓解这一点。
3. **早停（patience=20）** —— 论文中没有，是上游额外加的技巧。

### 7.8 上游结果

| 指标 | 值 |
|---|---|
| 最小验证损失 | 3.95 |
| 验证集 BLEU | 23.2 |
| 每轮耗时（秒） | 45 |
| 可训练参数量 | 17,519,828 |

<img src="https://user-images.githubusercontent.com/74488693/155708874-1ba0bb2a-c819-4cad-93fa-8d8545d261be.png" height="300" width="400">

### 7.9 上游结论

模型能利用注意力机制较好地学习句子上下文，能准确挑出源句中高频的主要词汇并正确翻译，但对低频词表现不佳。另外，语料含有大量特殊字符（`http: / /bugs.php.net /`、`satellite_exception_id()`）——因为它基于 PHP 脚本语言的文档。若在训练前去掉这些特殊字符，数据会丢失主要上下文，反而使训练更具挑战。

上游作者补充说明：受限于 GPU 资源，只能在小规模数据上训练，因此结果并不理想。

---

## 八、参考来源

1. [PHP Corpus Dataset](https://opus.nlpl.eu/PHP.php)
2. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et. al.
3. [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
4. 上游仓库：[shashankag14/Transformer-for-Machine-Translation](https://github.com/shashankag14/Transformer-for-Machine-Translation)

<p align="right">(<a href="#top">back to top</a>)</p>
