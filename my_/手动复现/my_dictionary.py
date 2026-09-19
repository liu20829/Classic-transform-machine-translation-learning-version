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
    return re.sub(r'\s+'," ",html.unescape(s).strip())
    pass

class Dictionary:
    def __init__(self,name):
        self.name=name
        self.word2index= {}#{token:索引}
        self.word2count={}#{token:频率}
        self.index2word={#{索引:token}给编号->token的解码器使用
            PAD_TOKEN:'PAD',SOS_TOKEN:'SOS',
            EOS_TOKEN:'EOS',UNK_TOKEN:'UNK'
        }
        self.n_count=N_SPECIAL # pad,SOS,ESO,UNK

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
        按mini_count裁掉低频词并重新分配编号

        遍历顺序取旧编号升序--旧编号本身就是“首次出现顺序”（add_word里按
        分配先后递增），所以裁剪后词的相对顺序不变，只是编号变紧凑了
        """
        if min_count<=1:
            return self
        kept=Dictionary(self.name)#新建一个字典类
        for word,_ in sorted(self.word2index.items(),key=lambda kv:kv):
            if self.word2count[word]>=min_count:
                kept.add_word(word)
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
        next(reader,None)
        # print(reader)


        for row in itertools.islice(reader,max_pairs):
            if len(row)<2:
                continue
            # print(row)
            # break
            zh=normalizeString(row[0])
            en=normalizeString(row[1])

            if not zh or not en:
                continue

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
    max_sent_len:序列最大长度
    min_count:最低频率数
    """

    src_sents,tgt_sents,dropped_long=load_pairs(csv_path,max_pairs,max_sent_len)
    print("句对：保留{}对，因超长丢弃{}对".format(len(src_sents),dropped_long))

    input_dic=Dictionary('zh')
    output_dic=Dictionary('en')

    for sentence in src_sents:
        input_dic.add_sentence(sentence)
    for sentence in tgt_sents:
        output_dic.add_sentence(sentence)

    raw_zh,raw_en=input_dic.n_count,output_dic.n_count#获取特殊字符数
    input_dic=input_dic.compact(min_count)
    output_dic=output_dic.compact(min_count)

    print("中文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_zh, input_dic.n_count, min_count))
    print("英文词表：{} -> {}（min_count={}，归入 UNK）".format(raw_en, output_dic.n_count, min_count))


    if save:
        save_dictionary(input_dic,input=True)
        save_dictionary(output_dic,input=False)

    return input_dic,output_dic,src_sents,tgt_sents

def save_dictionary(dictionary,input:bool=True):
    name='input_dic.pkl' if input is True else 'output_dic.pkl'
    with open(os.path.join(saved_chkpt,name),'wb') as f :
        pickle.dump(dictionary,f,pickle.HIGHEST_PROTOCOL)
    pass

if __name__ == "__main__":
    csv_dir=r"G:\PythonProject\follow-github-learn-agent\Transformer-for-Machine-Translation-main\my_data\WMT_dataset\wmt_zh_en_training_corpus.csv"
    load_pairs(csv_dir)

    input_dic, output_dic, src_sents, tgt_sents=create_dictionary(csv_dir)
    pass
