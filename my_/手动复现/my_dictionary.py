import csv
import os
import re,html
import pickle
import itertools

"""
中英翻译版词表构建。对应原版 dictionary.py，改造点有四：
  1. 数据源是两个纯文本文件 -> 一个 CSV（列 0 = 中文，列 1 = 英文），且必须跳表头
  2. 原版 normalizeString 的正则白名单只有 a-zA-Z.!?，会把汉字全删光，这里不再删字符
  3. 全量语料 2500 万句对读进内存会爆，加 max_pairs 限流
  4. 原版一个词只出现 1 次也占一个编号，这里按 min_count 裁剪，低频词归入新增的 UNK
"""

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3          # 新增：词表裁剪后，被裁掉的词统一用这个编号
N_SPECIAL = 4          # 特殊符号个数，普通词编号从 4 开始
saved_chkpt="save_chkpt"

def normalizeString(s):
    """
    HTML 实体解码 + 压空白。不删任何字符、不改大小写。

    原版这里是 unicodeToAscii + 两步正则(标点前插空格、非 [a-zA-Z.!?] 换成空格),
    那套规则针对的是未处理的捷克语原文。套到本语料上有三个问题:
      - 汉字不在白名单里,整句中文会被削成空白
      - 数据本身已经分好词("表演 的 明星"),再插空格会切出空串 token
      - 数据保留专有名词大小写(China / Mr),小写化会丢信息

    实体解码是必需的:发布方漏了这一步,&apos; &quot; 会以字面形式进词表,
    模型照着学出 "Berlin &apos;s" 这种输出。
    """
    return re.sub(r'\s+'," ",html.unescape(s).strip())
    pass

class Dictionary:
    def __init__(self,name):
        self.name=name
        # 三个字典各管一个方向,缺一不可:
        self.word2index= {}   # 词 -> 编号,编码用(tokenize 查它把句子变编号)
        self.word2count={}    # 词 -> 出现次数,裁剪用(compact 查它滤低频词)
        self.index2word={     # 编号 -> 词,解码用(detokenize 把编号变回词)
            PAD_TOKEN:'PAD',SOS_TOKEN:'SOS',
            EOS_TOKEN:'EOS',UNK_TOKEN:'UNK'
        }
        self.n_count=N_SPECIAL # 下一个可用的编号;也是词表总大小(含 4 个特殊符号)

        pass

    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)
        pass
    def add_word(self,word):
        if word not in self.word2index:#如果该token不存在token索引映射字典中
            self.word2index[word]=self.n_count#则创建该token索引
            self.word2count[word]=1#创建该token频率
            self.index2word[self.n_count]=word#创建索引token
            self.n_count+=1
        else:
            self.word2count[word]+=1
        pass
    def compact(self,min_count):
        """
        按 min_count 裁掉低频词并重新分配编号。

        期望的遍历顺序是旧编号升序——旧编号本身就是"首次出现顺序"(add_word 里按
        分配先后递增),这样裁剪后词的相对顺序不变,只是编号变紧凑了。
        """
        if min_count<=1:
            return self                      # 阈值 <=1 等于不裁,原样返回
        kept=Dictionary(self.name)           # 新建一个空词表,用来装裁剪后的词
        # ⚠️ key=lambda kv:kv 是按整个 (词,编号) 元组排 = 先按词排,不是按旧编号升序,
        #    跟上面 docstring 对不上。不会崩(重建后词表自洽),但编号不再是"首次出现顺序"。
        #    要按旧编号排得写 key=lambda kv: kv[1]
        for word,_ in sorted(self.word2index.items(),key=lambda kv:kv):
            if self.word2count[word]>=min_count:
                kept.add_word(word)
                # add_word 会把频次重置成 1,这里补回真实频次
                kept.word2count[word]=self.word2count[word]
        return kept

# 读预料：csv->成对的
def load_pairs(csv_path,max_pairs=10,max_sent_len:int=50):
    """
    返回（源句列表，目标句列表，因超长丢弃的对数）

    注意：max_pairs限制的是”读进来的行数“，超长过滤发生在其后
    所以最终句对数会略少于max_pairs,但两边表严格一一对应
    """
    src_sents,tgt_sents=[],[]
    dropped_long=0
    with open(csv_path,encoding='utf-8',newline='') as f:
        reader=csv.reader(f)
        next(reader,None)                # 首行是列名 '0,1',不是数据,跳过
        # print(reader)


        # islice 的第二个参数是 stop(最多吐几个元素),不是步长;
        # 传 None 表示不限制,会一路读到文件尾
        for row in itertools.islice(reader,max_pairs):
            if len(row)<2:
                continue                 # 残缺行(缺列),丢掉
            # print(row)
            # break
            zh=normalizeString(row[0])   # 列 0 = 中文
            en=normalizeString(row[1])   # 列 1 = 英文

            if not zh or not en:
                continue                 # 归一化后变成空的,丢掉

            #成对过滤：任一边超长就把正对丢掉，绝对不能让两个列表错位
            if len(zh.split(' '))> max_sent_len or len(en.split(' '))> max_sent_len:


                dropped_long+=1
                #此处之所以丢弃超过预定序列长度的是因为在批训练过程中序列长度必须等长，
                # 又因为不同语种的长度不一样如果简单的截断会导致较长的一方语言会被截断
                # 而对应的另一方如果比较短或者更加长的话在语义上就很那对齐
                continue

            src_sents.append(zh)
            tgt_sents.append(en)

    return src_sents,tgt_sents,dropped_long
    pass


# 建词表
def create_dictionary(csv_path,max_pairs=None,max_sent_len:int=50,min_count:int=2,save:bool=True):
    """
    csv_path     : 语料 CSV 路径(列 0 = 中文,列 1 = 英文)

    max_pairs    : 最多读多少行。None = 不限制,会读完整份文件
    max_sent_len : 句子最多多少 token,任一边超长就整对丢弃;
                   同时决定 tokenize 之后的序列长度(= 它 + 2)
    min_count    : 出现次数低于此值的词不单独占编号,统一归入 UNK
    save         : 是否把词表 pickle 落盘到 saved_chkpt/
    """

    # ⚠️ max_pairs 默认 None,会一路传到 islice(reader, None) = 读到文件尾,
    #    也就是啃完整份 6.3GB。调用方务必显式传值。
    src_sents,tgt_sents,dropped_long=load_pairs(csv_path,max_pairs,max_sent_len)
    print("句对：保留{}对，因超长丢弃{}对".format(len(src_sents),dropped_long))

    input_dic,output_dic=_count(src_sents,tgt_sents,min_count)

    if save:
        save_dictionary(input_dic,input=True)
        save_dictionary(output_dic,input=False)

    return input_dic,output_dic,src_sents,tgt_sents


def _count(src_sents,tgt_sents,min_count):
    """
    从句子列表统计词频并裁剪,返回 (input_dic, output_dic)。

    单独抽出来是为了让"现建"和"读缓存"两条路能共用同一段逻辑,
    也避免缓存未命中时把 CSV 读两遍。
    """
    input_dic=Dictionary('zh')      # 源语言(中文)
    output_dic=Dictionary('en')     # 目标语言(英文)

    # 逐句喂进去统计词频。这一步结束后,两个词表各自拿到了全量词汇和出现次数
    for sentence in src_sents:
        input_dic.add_sentence(sentence)
    for sentence in tgt_sents:
        output_dic.add_sentence(sentence)

    # 记下裁剪前的大小,只为了下面 print 出"裁剪前 -> 裁剪后"
    raw_zh,raw_en=input_dic.n_count,output_dic.n_count
    input_dic=input_dic.compact(min_count)      # 裁低频词,返回的是新词表
    output_dic=output_dic.compact(min_count)

    print("中文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_zh, input_dic.n_count, min_count))
    print("英文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_en, output_dic.n_count, min_count))
    return input_dic,output_dic


def load_or_create(csv_path,max_pairs,max_sent_len,min_count,chkpt=None):
    """
    返回 (input_dic, output_dic, src_sents, tgt_sents)。

    词表走缓存,句子不缓存(全量 2500 万句对存下来是 GB 级)。
    所以无论命中与否,CSV 都只读一遍 —— 句子反正要读,
    命中缓存省的只是"统计词频 + 裁剪"那一步。

    缓存文件名带上了三个建表参数: 参数一变文件名就变,自动重建,
    避免"改了 min_count 却还在用旧词表"这种不报错的错误。
    """
    chkpt=saved_chkpt if chkpt is None else chkpt
    os.makedirs(chkpt,exist_ok=True)
    tag='{}_{}_{}'.format(max_pairs,max_sent_len,min_count)
    ip=os.path.join(chkpt,'input_dic_{}.pkl'.format(tag))
    op=os.path.join(chkpt,'output_dic_{}.pkl'.format(tag))

    src_sents,tgt_sents,dropped_long=load_pairs(csv_path,max_pairs,max_sent_len)
    print("句对：保留{}对，因超长丢弃{}对".format(len(src_sents),dropped_long))

    if os.path.exists(ip) and os.path.exists(op):
        with open(ip,'rb') as f: input_dic=pickle.load(f)
        with open(op,'rb') as f: output_dic=pickle.load(f)
        print('加载缓存词表：',os.path.basename(ip))
        return input_dic,output_dic,src_sents,tgt_sents

    # 没缓存:现建,然后落盘。
    # 落盘在这里(模块作用域)而不是 __main__ 里,类路径才是 my_dictionary.Dictionary,
    # 换个文件 pickle.load 才找得到。
    input_dic,output_dic=_count(src_sents,tgt_sents,min_count)
    with open(ip,'wb') as f: pickle.dump(input_dic,f,pickle.HIGHEST_PROTOCOL)
    with open(op,'wb') as f: pickle.dump(output_dic,f,pickle.HIGHEST_PROTOCOL)
    print('新建词表并缓存：',os.path.basename(ip))
    return input_dic,output_dic,src_sents,tgt_sents

def save_dictionary(dictionary,input:bool=True):
    # 目录 saved_chkpt 是相对路径,落到哪取决于当前工作目录
    # ⚠️ 从 __main__ 里调会 pickle 成 __main__.Dictionary,
    #    别的模块加载时报 Can't get attribute 'Dictionary'
    name='input_dic.pkl' if input is True else 'output_dic.pkl'
    with open(os.path.join(saved_chkpt,name),'wb') as f :
        pickle.dump(dictionary,f,pickle.HIGHEST_PROTOCOL)
    pass

if __name__ == "__main__":
    csv_dir=r"G:\PythonProject\follow-github-learn-agent\Transformer-for-Machine-Translation-main\my_data\WMT_dataset\wmt_zh_en_training_corpus.csv"

    # 单独跑一次,看读到了什么(默认只读 10 行)
    load_pairs(csv_dir)

    # ⚠️ 两点要注意:
    #   1. max_pairs 必须显式传,默认 None 会读完整份 6.3GB
    #   2. save=False —— 从 __main__ 里存,类会被记成 __main__.Dictionary,
    #      别的模块 pickle.load 时报 Can't get attribute 'Dictionary'。
    #      要落盘缓存,请从导入方(如 my_train.py)调 load_or_create()
    input_dic, output_dic, src_sents, tgt_sents=create_dictionary(
        csv_dir, max_pairs=200000, save=False)
    pass
