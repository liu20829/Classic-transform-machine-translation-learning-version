"""
中英翻译版词表构建。对应原版 dictionary.py，改造点有四：
  1. 数据源是两个纯文本文件 -> 一个 CSV（列 0 = 中文，列 1 = 英文），且必须跳表头
  2. 原版 normalizeString 的正则白名单只有 a-zA-Z.!?，会把汉字全删光，这里不再删字符
  3. 全量语料 2500 万句对读进内存会爆，加 max_pairs 限流
  4. 原版一个词只出现 1 次也占一个编号，这里按 min_count 裁剪，低频词归入新增的 UNK
"""

import csv
import html
import itertools
import os
import pickle
import re

import utils_ch_en as utils

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3          # 新增：词表裁剪后，被裁掉的词统一用这个编号
N_SPECIAL = 4          # 特殊符号个数，普通词编号从 4 开始

# ########################################################################
# # 文本预处理
# ########################################################################
def normalizeString(s):
    """
    两步：HTML 实体解码 + 压空白。不删任何字符、不改大小写。

    原版这里是 unicodeToAscii + 两步正则（标点前插空格、非 [a-zA-Z.!?] 换成空格）。
    那套规则针对的是未处理的捷克语原文，套到本语料上有三个问题：
      - 汉字不在白名单里，整句中文会被削成空白
      - 数据本身已经分好词（"表演 的 明星"），再插空格会产生连续空格，
        split(' ') 就会切出空串 token
      - 数据保留专有名词大小写（China / Mr / God 共占 5.4% 的 token），小写化会丢信息

    实体解码是必需的：发布方漏了这一步，&apos; &quot; &amp; 以字面形式出现在
    1.56% 的 token 里（如 "Berlin &apos;s"），不解码模型会照着学出 &apos; 这种输出。
    """
    return re.sub(r'\s+', ' ', html.unescape(s).strip())


# ########################################################################
# # DICTIONARY
# ########################################################################
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}    # mapping word to its idx
        self.word2count = {}    # frequency of each word
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS",
                           EOS_TOKEN: "EOS", UNK_TOKEN: "UNK"}
        self.n_count = N_SPECIAL    # pad, SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1

    def compact(self, min_count):
        """
        按 min_count 裁掉低频词并重新分配编号。

        遍历顺序取旧编号升序——旧编号本身就是"首次出现顺序"（add_word 里按
        分配先后递增），所以裁剪后词的相对顺序不变，只是编号变紧凑了。
        """
        if min_count <= 1:
            return self

        kept = Dictionary(self.name)
        for word, _ in sorted(self.word2index.items(), key=lambda kv: kv[1]):
            if self.word2count[word] >= min_count:
                kept.add_word(word)
                kept.word2count[word] = self.word2count[word]   # add_word 会重置成 1，还原真实频次
        return kept


# ########################################################################
# # 读语料：CSV -> 成对的 (中文句, 英文句)
# ########################################################################
def load_pairs(csv_path, max_pairs=None, max_sent_len=50):
    """
    返回 (源句列表, 目标句列表, 因超长丢弃的对数)。

    注意：max_pairs 限制的是"读进来的行数"，超长过滤发生在其后，
    所以最终句对数会略少于 max_pairs，但两边下标严格一一对应。
    """
    src_sents, tgt_sents = [], []
    dropped_long = 0

    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)      # 首行是列名 '0,1'，不是数据

        for row in itertools.islice(reader, max_pairs):
            if len(row) < 2:
                continue        # 残缺行，丢掉
            zh = normalizeString(row[0])
            en = normalizeString(row[1])
            if not zh or not en:
                continue

            # 成对过滤：任一边超长就把整对丢掉，绝不能让两个列表错位
            if len(zh.split(' ')) > max_sent_len or len(en.split(' ')) > max_sent_len:
                dropped_long += 1
                continue

            src_sents.append(zh)
            tgt_sents.append(en)

    return src_sents, tgt_sents, dropped_long


# ########################################################################
# # 建词表
# ########################################################################
def create_dictionary(csv_path, max_pairs=None, max_sent_len=50, min_count=2, save=True):
    src_sents, tgt_sents, dropped_long = load_pairs(csv_path, max_pairs, max_sent_len)
    print("句对：保留 {} 对，因超长丢弃 {} 对".format(len(src_sents), dropped_long))

    input_dic = Dictionary('zh')
    output_dic = Dictionary('en')

    for sentence in src_sents:
        input_dic.add_sentence(sentence)
    for sentence in tgt_sents:
        output_dic.add_sentence(sentence)

    raw_zh, raw_en = input_dic.n_count, output_dic.n_count
    input_dic = input_dic.compact(min_count)
    output_dic = output_dic.compact(min_count)
    print("中文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_zh, input_dic.n_count, min_count))
    print("英文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_en, output_dic.n_count, min_count))

    # save=False 供本文件的 __main__ 自检用：直接跑脚本时 Dictionary 会被 pickle 成
    # __main__.Dictionary，别的模块反序列化会报 "Can't get attribute 'Dictionary'"。
    # 正常流程由 train_ch_en.py 导入后调用，类名带模块前缀，没有这个问题。
    if save:
        save_dictionary(input_dic, input=True)
        save_dictionary(output_dic, input=False)

    return input_dic, output_dic, src_sents, tgt_sents


def save_dictionary(dictionary, input=True):
    name = 'input_dic.pkl' if input is True else 'output_dic.pkl'
    with open(os.path.join(utils.saved_chkpt, name), 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    input_dic, output_dic, src_sents, tgt_sents = create_dictionary(
        utils.data_path, utils.max_pairs, utils.max_sent_len, utils.min_count, save=False)

    # ---- 1. 特殊符号编号 ----
    assert [input_dic.index2word[i] for i in range(N_SPECIAL)] == ['PAD', 'SOS', 'EOS', 'UNK']
    assert [output_dic.index2word[i] for i in range(N_SPECIAL)] == ['PAD', 'SOS', 'EOS', 'UNK']

    # ---- 2. 编号连续无空洞，且不重号 ----
    assert sorted(input_dic.index2word) == list(range(input_dic.n_count)), '中文词表编号有空洞'
    assert sorted(output_dic.index2word) == list(range(output_dic.n_count)), '英文词表编号有空洞'
    assert len(input_dic.word2index) + N_SPECIAL == input_dic.n_count

    # ---- 3. 裁剪生效：不该再有低于阈值的词 ----
    assert all(c >= utils.min_count for c in input_dic.word2count.values()), '中文词表还有低频词'
    assert all(c >= utils.min_count for c in output_dic.word2count.values()), '英文词表还有低频词'

    # ---- 4. 列序没搞反：源句是中文，目标句是英文 ----
    cjk = re.compile(r'[一-鿿]')
    assert any(cjk.search(w) for w in src_sents[0].split(' ')), '源句里找不到汉字，列序反了？'
    assert not any(cjk.search(w) for w in tgt_sents[0].split(' ')), '目标句里有汉字，列序反了？'

    # ---- 5. 配对校验：中英必须一一对应 ----
    # 错位了训练照跑不误，但学出来全是垃圾且不报错，所以这条必须断言
    assert len(src_sents) == len(tgt_sents), '中英句数不等，配对已错位'
    assert src_sents[0].startswith('表演') and tgt_sents[0].startswith('the show'), \
        '第一对不是预期的译文对：\n  zh={!r}\n  en={!r}'.format(src_sents[0], tgt_sents[0])

    # ---- 6. HTML 实体确实解码了（转义形式消失，解码形式在场）----
    assert '&apos;' not in output_dic.word2index, 'HTML 实体没解码'
    assert "'s" in output_dic.word2index, "解码后的 's 不在词表里，unescape 没生效？"

    print()
    print('抽样 5 对（人工核对语义是否对得上）:')
    for i in (0, 1, 2, 1000, 5000):
        print('  [{}] zh: {}'.format(i, src_sents[i][:46]))
        print('      en: {}'.format(tgt_sents[i][:46]))
    print()
    print('词表自检通过')
