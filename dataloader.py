"""
@author : Shashank Agarwal
@when : 08-12-2021
@homepage : https://github.com/shashankag14
"""
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import utils

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
def get_dataloader(src_tokens, tgt_tokens) :
    # 1. Split the SRC and TGT tokens into train, valid and test sets
    train_src, remain_src = train_test_split(src_tokens, test_size=0.2,
                                             random_state=utils.args.seed)
    train_tgt, remain_tgt = train_test_split(tgt_tokens, test_size=0.2,
                                             random_state=utils.args.seed)

    valid_src, test_src = train_test_split(remain_src, test_size=0.2,
                                           random_state=utils.args.seed)
    valid_tgt, test_tgt = train_test_split(remain_tgt, test_size=0.2,
                                           random_state=utils.args.seed)

    # 2. Create lists of tran/valid/test source-target tuples
    train_data = list(zip(train_src, train_tgt))
    valid_data = list(zip(valid_src, valid_tgt))
    test_data = list(zip(test_src, test_tgt))

    # 3. Create custom dataset using the above lists for Train/Valid/Test
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    test_dataset = CustomDataset(test_data)
    print("Number of sentences in Train/Valid/Test :", len(train_dataset), len(valid_dataset), len(test_dataset))

    # 4. Create batch iterator for the Train/Valid/Test dataloaders
    # 用标准 DataLoader 替代 torchtext.legacy 的 BucketIterator(BucketIterator 已随 torchtext 0.17 移除)
    # collate_fn 原样返回,保持 batch 元素为 dict 列表,训练/评估循环的写法不用改
    train_dataloader = DataLoader(train_dataset, batch_size=utils.batch_size,
                                  shuffle=True, collate_fn=lambda x: x)
    valid_dataloader = DataLoader(valid_dataset, batch_size=utils.batch_size,
                                  shuffle=False, collate_fn=lambda x: x)

    return train_dataloader, valid_dataloader, test_dataset

