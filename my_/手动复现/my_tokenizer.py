"""
中英翻译版编码/解码。对应原版 tokenizer.py，改造点有三：
  1. 查词表改用 .get(word, UNK_TOKEN)——词表按 min_count 裁剪后必然出现 OOV，
     原版的 word2index[word] 直接下标会 KeyError（实测 UNK 命中率约 1.8%）
  2. 长度做截断保护——全流程依赖"所有序列等长"，一旦不等长，
     train.py 的 torch.cat 会直接崩
  3. detokenize 保留 UNK（原版只滤 PAD/SOS/EOS）
"""
import my_dictionary as dict

def tokenize(sentence,dictionary,MAX_LENGTH=50):
    """
    sentence:传进来的句子
    句子——>编号列表，长度恒为MAX_LENGTH+2
    MAX_LENGTH+2不是笔误：原版写的是[SOS]+n词+[EOS]+PAD*(MAX_LENGTH)
    填PAD时把SOS/EOS自己算漏了，实际长度为2，这是个off-by-2,但所有句子
    一律等长，不影响训练，所以保持原行为，别按MAX_LENGTH去堆形状

    超长句子直接截断：训练测的batch是torch.cat拼起来的，长度必须完全一致
    词表里查不到的词映到UNK_TOKEN
    """
    words=sentence.split(' ')#
    # print(f"tokenize传入的句子{words}")   # 注:语料含 € 等 GBK 编不出的字符,直接 print 会 UnicodeEncodeError
    if len(words)>MAX_LENGTH:
        words=words[:MAX_LENGTH]

    token=[dict.SOS_TOKEN]#获取sos的id
    token+=[dictionary.word2index.get(word,dict.UNK_TOKEN) for word in words]
    token.append(dict.EOS_TOKEN)
    token+=[dict.PAD_TOKEN]*(MAX_LENGTH-len(words))
    # print(len(token))
    return token
    pass

def detokenize(x,vocab):
    """
    编号——>字符串。滤掉PAD/SOS/EOS 保留UNK

    cunk,不滤掉是可以的，模型暑促UNK 说明它在这里没学会滤掉会让翻译结果
    看起来比实际干净，保留后BLEU两侧都能看到UNK，也算得一致

    """
    words=[]
    for i in x:
        word=vocab.index2word[i]
        if word!='EOS' and word !='SOS' and word!='PAD':
            words.append(word)
    # print(' '.join(words))   # 同 tokenize:原文含 GBK 编不出的字符会崩
    return ' '.join(words)
    pass
if __name__=="__main__":
    MAX_LEN = 50

    # ---- 用一个临时小词表做单元检查，不依赖 CSV ----
    d = dict.Dictionary('test')
    d.add_sentence('你好 世界 你好 世界')
    assert d.word2index['你好'] == dict.N_SPECIAL, '头一个词应该拿到第一个普通编号'

    # ---- 1. 长度恒定 ----
    for n in (1, 5, 20, MAX_LEN):
        sent = ' '.join(['词'] * n)
        assert len(tokenize(sent, d, MAX_LEN)) == MAX_LEN + 2, '长度不是 MAX_LENGTH+2'

    # ---- 2. 头部 SOS / 尾部 PAD 的位置 ----
    tok = tokenize('你好 世界', d, MAX_LEN)
    assert tok[0] == dict.SOS_TOKEN
    assert tok[1] == d.word2index['你好'] and tok[2] == d.word2index['世界']
    assert tok[3] == dict.EOS_TOKEN
    assert all(t == dict.PAD_TOKEN for t in tok[4:])

    # ---- 3. OOV 返回 UNK 而不是 KeyError ----
    oov = '这个词肯定不在表里xyz'
    assert oov not in d.word2index
    tok_oov = tokenize('你好 {} 世界'.format(oov), d, MAX_LEN)
    assert tok_oov[2] == dict.UNK_TOKEN, 'OOV 没有映到 UNK'

    # ---- 4. 超长截断，长度仍然恒定 ----
    tok_long = tokenize(' '.join(['词'] * (MAX_LEN + 30)), d, MAX_LEN)
    assert len(tok_long) == MAX_LEN + 2, '截断后长度变了'

    # ---- 5. 往返一致（UNK 除外，它还原不成原词）----
    for sent in ['你好 世界', '你好 世界 你好 世界']:
        print(sent)
        assert detokenize(tokenize(sent, d, MAX_LEN), d) == sent, '往返不一致: ' + sent

    # ---- 6. detokenize 滤 PAD/SOS/EOS，但保留 UNK ----
    assert detokenize([dict.PAD_TOKEN, dict.SOS_TOKEN, dict.UNK_TOKEN,
                        d.word2index['你好'], dict.EOS_TOKEN, dict.PAD_TOKEN], d) == 'UNK 你好'

    print('长度恒为     :', len(tokenize('你好 世界', d, MAX_LEN)))
    print('OOV 映到编号 :', tok_oov[2], '(UNK_TOKEN={})'.format(dict.UNK_TOKEN))
    print('往返结果     :', repr(detokenize(tokenize('你好 世界', d, MAX_LEN), d)))
    print('分词器自检通过')
    pass