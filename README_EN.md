<div id="top"></div>

# Classic Transformer Machine Translation — Learning Repo

> A from-scratch Transformer implementation (upstream) + my hand-written reproduction, empirical tests, and study notes
> 中文版：[README.md](README.md)

---

## 1. What This Repo Is

| | |
|---|---|
| **Code origin** | [shashankag14/Transformer-for-Machine-Translation](https://github.com/shashankag14/Transformer-for-Machine-Translation), a from-scratch PyTorch Transformer based on [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) |
| **Task** | Czech → English machine translation (PHP corpus) |
| **What I did** | Hand-written reproduction, empirical verification, study notes |
| **Changes to upstream** | **No logic changed** — only Chinese comments added to `model/attention.py` and `model/encoder.py` |

The point of this repo in one line:

> **Upstream gives you an implementation that runs; what I did is take it apart.**

The repo name ("learning class") reflects that goal. This is not about chasing BLEU scores — it is about pinning down every tensor shape, every memory movement, and every "why is it written this way". Whenever something can be measured, I measure it rather than reason it out.

---

## 2. Directory Map

| Path | Contents | Origin |
|---|---|---|
| `model/` | The five Transformer modules: `attention.py`, `encoder.py`, `decoder.py`, `position_encoding.py`, `transformer.py` | Upstream |
| `train.py` / `translate.py` / `plot.py` | Training, inference, plotting entry points | Upstream |
| `tokenizer.py` / `dictionary.py` / `dataloader.py` / `utils.py` / `bleu_metric.py` | Data pipeline and evaluation | Upstream |
| `scripts/` | Data download, environment setup, vocab export | Upstream |
| `data/cs-en/` | Czech–English parallel corpus | Data |
| `saved_chkpt/` | Training output: `best_model.pt` and vocabularies | Output |
| `results/` | Loss curves, loss logs, BLEU | Output |
| **`my_/手动复现/`** | My hand-written encoder reproduction, with smoke test and mask verification | **Mine** |
| **`my_/测试/`** | Empirical scripts: shape tracing, memory behaviour | **Mine** |
| **`my_/transformer网络结构图/`** | Architecture diagrams | **Mine** |
| **`my_doc/`** | Study notes (plus an interactive architecture diagram) | **Mine** |
| `Attention-is-all-you-need/` | The paper (PDF) | Reference |

---

## 3. Quick Start

### 3.1 Environment

```bash
sh scripts/requirements.sh
```

### 3.2 Data

The corpus is already in `data/cs-en/` (`cs.txt` / `en.txt`). If missing, re-download with the upstream script — note you must extract `PHP.cs-en.cs` and `PHP.cs-en.en` into `data/`:

```bash
sh scripts/download_data.sh
```

### 3.3 Train

```bash
python train.py
```

With explicit hyperparameters (full table in §7.6):

```bash
python train.py --epoch 150 --batch_size 32 --d_model 256 --n_layers 7 \
                --n_heads 8 --ffn_hidden 1024 --dropout 0.15 \
                --max_sent_len 50 --init_lr 1e-4
```

### 3.4 Plot the training curve

Can be run at any time during training; the plot is saved to `results/`:

```bash
python plot.py
```

### 3.5 Translate with the trained weights

```bash
python translate.py
```

Output goes to `results/translation_results.txt`. `translate.py` loads `saved_chkpt/best_model.pt`.

---

## 4. My Learning Path

In the order I actually studied:

| Stage | Topic | Output |
|---|---|---|
| ① | **Positional encoding** — from the `sin/cos` formula to "how addition becomes rotation" | 2 notes + 1 interactive frequency explorer |
| ② | **Multi-head attention** — how shapes transform, and whether the underlying data actually moves | 1 note + 2 empirical scripts |
| ③ | **Decoder** — its input/output form, why training needs no loop while inference must loop | 2 notes + an interactive architecture diagram |
| ④ | **Hand-written reproduction** — rewrite the encoder from my own understanding, without looking at the source | `my_/手动复现/transformer_encoder.py` |

One principle throughout: **measure it, don't reason it out.**

---

## 5. Reproduction and Empirical Tests

### 5.1 Hand-written reproduction (`my_/手动复现/`)

| File | Status | Contents |
|---|---|---|
| `transformer_encoder.py` | ✅ 252 lines | Encoder rewritten from my own understanding: `feedforwardlayer`, `MultiHeadAttention`, `transformer_encoder_layer`, `positional`, `embedding_positional`, `transformer_encoder` |
| `transformer_decoder.py` | 🚧 empty | To be done |

`transformer_encoder.py` ships with a self-check in `__main__`. Just run it:

```bash
python my_/手动复现/transformer_encoder.py
```

The self-check does two things:

1. **Smoke test** — build a dummy input (with PAD), assert the output shape is correct and contains no NaN
2. **Verify the mask actually works** — change the words at masked positions and observe that real positions are unaffected (measured `diff_real = 0.0`) while masked positions change (`diff_pad = 3.85`); then pass `mask=None` and the real positions change immediately (`diff_nomask = 0.82`)

### 5.2 Empirical scripts (`my_/测试/`)

| Script | Lines | What it measures |
|---|---|---|
| `shape_trace.py` | 71 | Runs `scaled_dot_product_attn` on the **real project code**, printing the tensor shape at each step and checking the softmax behaviour of PAD rows |
| `view_permute_memory.py` | 40 | Measures the memory behaviour of `view` / `permute`: both only change strides, never move data (`data_ptr` unchanged); only `.contiguous()` actually copies |
| `bb.py` | 19 | Positional-encoding formula experiment |
| `aa.py` | 5 | Scratch |

Run them with your local Python:

```bash
python my_/测试/shape_trace.py
python my_/测试/view_permute_memory.py
```

One result in `shape_trace.py` **contradicts intuition**: rows where PAD is the query do *not* end up with a uniform distribution — the weights sum to 1.0 over real keys and 0.0 over padding keys. That came out of a measurement, not a derivation.

---

## 6. Study Notes (`my_doc/`)

### 6.1 Notes

| Note | Lines | Topic |
|---|---|---|
| `20260907_位置编码-从加法到旋转.md` | 497 | Derives the full positional-encoding chain starting from a 2-D vector as the minimal example |
| `20260907_算例-加位置编码的变换步骤.md` | 164 | How the position fingerprint is generated step by step in code at 4-D, and the full "add to word vector → dot product" chain |
| `多头注意力的张量形状与内存布局-知识概念笔记.md` | 379 | How shapes transform → whether the data moves → why the transform is needed → what "similarity" actually means |
| `Transformer解码器的输入输出形式-知识概念笔记.md` | 804 | What the decoder eats, what it emits, why it's sliced that way; organised along a question chain, with paper excerpts and exposure bias |
| `Transformer解码器理解路径-RNN对照版.md` | 787 | The same topic told a different way: "what would someone coming from RNNs assume?" |

The last two are **two takes on the same subject** and cross-reference each other: the first is organised by question chain, the second by learning path.

### 6.2 Interactive artefacts

| File | Description |
|---|---|
| `transformer-架构图/index.html` | Interactive architecture diagram: an overview plus three sub-diagrams (encoder / decoder / attention). Single-click a node to jump to its code explanation, double-click a dashed node to expand a sub-diagram. Just open it in a browser |
| `pe-frequency-explorer.html` | Positional-encoding frequency explorer: visualises the wavelength distribution across dimensions |

---

## 7. Upstream Project Notes

> Reproduced from the upstream README for traceability.

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

### 7.5 Dataset

PHP Corpus Czech-English

| Item | Value |
|---|---|
| Total sentences in corpus | ~33,000 |
| Unique sentences with <50 words | 5464 |
| Train dataset | 4371 |
| Validation dataset | 874 |
| Test dataset | 219 |
| Source (Czech) vocabulary size | 8891 |
| Target (English) vocabulary size | 4564 |

### 7.6 Hyperparameters (`train.py`)

| Parameter | Description | Value used | Value in the paper |
| --- | --- | --- | --- |
| `--epoch` | Number of epochs | 150 | N/A |
| `--batch_size` | Batch size | 32 | N/A |
| `--d_model` | Word embedding size | 256 | 512 |
| `--n_layers` | Number of enc/dec layers | 7 | 6 |
| `--n_heads` | Number of attention heads | 8 | 8 |
| `--ffn_hidden` | Hidden units in FFN | 1024 | 2048 |
| `--dropout` | Dropout probability | 0.15 | 0.1 |
| `--max_sent_len` | Max sentence length | 50 | N/A |
| `--init_lr` | Initial learning rate | 1e-4 | N/A |
| `--scheduler_factor` | LR decay factor | 0.9 | 0.9 |
| `--optim_adam_eps` | Adam epsilon | 5e-9 | 1e-9 |
| `--optim_patience` | Epochs before LR decay | 8 | N/A |
| `--optim_warmup` | Optimizer warmup | 16000 | 4000 |
| `--optim_weight_decay` | Weight decay | 5e-4 | N/A |
| `--clip` | Gradient clipping threshold | 1.0 | N/A |
| `--seed` | Random seed | 1111 | N/A |
| `--label_smooth_eps` | Label smoothing | 0.1 | 0.1 |
| `--early_stop_patience` | Early-stopping patience | 20 | N/A |

### 7.7 Regularization

Following the paper[2], two regularization techniques are active **only during training**:

1. **Residual Dropout (0.15)** — applied to the embeddings (positional + word) and to the output of every sublayer in the encoder and decoder. `nn.Dropout()` is used rather than `nn.functional.dropout` so that it deactivates automatically under `eval()`.
2. **Label Smoothing (eps=0.1)** — one-hot labels encourage the largest possible logit gaps, making the model over-confident and prone to overfitting; label smoothing mitigates this by encouraging smaller logit gaps.
3. **Early Stopping (patience=20)** — not in the paper; an extra technique added upstream.

### 7.8 Upstream Results

| Statistic | Value |
|---|---|
| Minimum validation loss | 3.95 |
| Validation set BLEU | 23.2 |
| Time per epoch (seconds) | 45 |
| Trainable parameters | 17,519,828 |

<img src="https://user-images.githubusercontent.com/74488693/155708874-1ba0bb2a-c819-4cad-93fa-8d8545d261be.png" height="300" width="400">

### 7.9 Upstream Conclusion

The model is able to exploit the attention mechanism to learn the context of the sentence well, as it can pick the main high-frequency words from the source sentence and translate them correctly. However, it fails to perform well on low-frequency words. Another observation is that the dataset contains a lot of special characters (`http: / /bugs.php.net /`, `satellite_exception_id()`) since it is based on a guide to the PHP scripting language. After preprocessing these sentences and removing the special characters before training, the data tends to lose its main context, which makes training more challenging.

Upstream also notes that due to limited GPU availability, the model could only be trained on very little data, so the results are not satisfactory.

---

## 8. References

1. [PHP Corpus Dataset](https://opus.nlpl.eu/PHP.php)
2. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et. al.
3. [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
4. Upstream repo: [shashankag14/Transformer-for-Machine-Translation](https://github.com/shashankag14/Transformer-for-Machine-Translation)

<p align="right">(<a href="#top">back to top</a>)</p>
