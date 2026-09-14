"""
中英翻译版数据切分。对应原版 dataloader.py，逻辑本身与语言无关，只改了 import。

这个文件真正值得留意的是一件事：源句和目标句**凭什么能按位置 zip 起来**。
原版没写、也没断言，但它成立的前提很窄——两次 train_test_split 用同一个
random_state，且两个列表等长，permutation 才会完全一致。任一处改动，
配对就会静默错位，训练照跑不误、学出来全是垃圾。
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import utils_ch_en as utils

# ########################################################################
# # CUSTOM DATASET - Ignores the sentences longer than MAX_LEN and
# #                 returns tuples of Source and Target
# ########################################################################
class CustomDataset(object):
    def __init__(self, data):
        self.src_data = []
        self.tgt_data = []
        for src,tgt in data:
            self.src_data.append(src)
            self.tgt_data.append(tgt)
        self.num_examples = len(self.src_data)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        return {'src': self.src_data[item], 'tgt': self.tgt_data[item]}

# ########################################################################
# # Method for splitting the tokens and creating dataloaders for train, valid and test
# ########################################################################
def get_dataloader(src_tokens, tgt_tokens, batch_size=None, seed=None, collate=None):
    # batch_size / seed 默认取 utils_ch_en 的配置；显式传入便于自检时用小组件
    batch_size = utils.batch_size if batch_size is None else batch_size
    seed = utils.args.seed if seed is None else seed

    # 1. Split the SRC and TGT tokens into train, valid and test sets
    #    ⚠️ 下面四次切分必须用**同一个 seed**，且 src_tokens / tgt_tokens 等长。
    #    切分的本质是 permutation(n)，seed 相同 + n 相同 -> 下标映射完全一致，
    #    所以后面 zip 出来的才是正确的译文对。
    #    注意比例不是 8:1:1——是先切 20% 出去，再从剩下的里切 20%，
    #    最终 train:valid:test = 0.8 : 0.16 : 0.16。
    train_src, remain_src = train_test_split(src_tokens, test_size=0.2, random_state=seed)
    train_tgt, remain_tgt = train_test_split(tgt_tokens, test_size=0.2, random_state=seed)

    valid_src, test_src = train_test_split(remain_src, test_size=0.2, random_state=seed)
    valid_tgt, test_tgt = train_test_split(remain_tgt, test_size=0.2, random_state=seed)

    # 2. Create lists of train/valid/test source-target tuples
    train_data = list(zip(train_src, train_tgt))
    valid_data = list(zip(valid_src, valid_tgt))
    test_data = list(zip(test_src, test_tgt))

    # 3. Create custom dataset using the above lists for Train/Valid/Test
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    test_dataset = CustomDataset(test_data)
    print("Number of sentences in Train/Valid/Test :", len(train_dataset), len(valid_dataset), len(test_dataset))

    # 4. Create batch iterator for the Train/Valid/Test dataloaders
    # collate_fn 原样返回，batch 元素是 dict 的列表，堆成张量的活由训练脚本做
    # （原版从 torchtext BucketIterator 迁移时的接法，保持一致）
    if collate is None:
        collate = lambda x: x

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate)

    return train_dataloader, valid_dataloader, test_dataset


if __name__ == "__main__":
    # ---- 用"带编号的假数据"直接检验配对没被打乱 ----
    # 关键：src 和 tgt 的编号必须一一对应（s3 只能配 t3），
    # 用 range 当假数据是测不出来的（两边数值一样，错位了也看不出来）
    n = 1000
    src_tokens = ['s{}'.format(i) for i in range(n)]
    tgt_tokens = ['t{}'.format(i) for i in range(n)]

    train_loader, valid_loader, test_dataset = get_dataloader(
        src_tokens, tgt_tokens, batch_size=32, seed=1111)

    def suffix(x):
        return int(x[1:])

    # 1. 数量守恒
    total = len(train_loader.dataset) + len(valid_loader.dataset) + len(test_dataset)
    assert total == n, '切分后总数对不上: {} != {}'.format(total, n)

    # 2. 三份集合两两无交集
    tr = {suffix(s) for s in train_loader.dataset.src_data}
    va = {suffix(s) for s in valid_loader.dataset.src_data}
    te = {suffix(s) for s in test_dataset.src_data}
    assert not (tr & va) and not (tr & te) and not (va & te), 'train/valid/test 有重叠'

    # 3. 配对没被打乱：src 的编号和 tgt 的编号必须处处相同
    for name, ds in [('train', train_loader.dataset), ('valid', valid_loader.dataset),
                     ('test', test_dataset)]:
        bad = [(s, t) for s, t in zip(ds.src_data, ds.tgt_data) if suffix(s) != suffix(t)]
        assert not bad, '{} 集配对错位，前几例: {}'.format(name, bad[:3])

    # 4. 一个 batch 的形状与元素结构
    batch = next(iter(train_loader))
    assert isinstance(batch, list) and len(batch) == 32, 'batch 结构不是 dict 列表'
    assert set(batch[0].keys()) == {'src', 'tgt'}, 'batch 元素字段不对'

    print('切分总数     :', total, '(守恒)')
    print('集合无重叠   : True')
    print('配对未错位   : True')
    print('一个 batch   : {} 个 dict，字段 {}'.format(len(batch), sorted(batch[0].keys())))
    print('数据加载器自检通过')
