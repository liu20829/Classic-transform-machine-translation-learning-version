"""
导出 pkl 格式词表为可读 JSON。

背景:Dictionary 用 pickle 二进制落盘(saved_chkpt/*_dic.pkl),文本编辑器打开是乱码,
     而词表本质只是 token->id 的纯映射,JSON 更直观且可跨语言使用。

用法:在项目根目录运行  python scripts/export_vocab.py
输出:saved_chkpt/input_dic.json(捷克语)、saved_chkpt/output_dic.json(英语)
"""
import json
import os
import pickle
import sys

# pickle 反序列化 Dictionary 对象时需要 import dictionary 模块,把项目根挂进搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (pkl 路径, 导出 json 路径)
FILES = [
    ("saved_chkpt/input_dic.pkl", "saved_chkpt/input_dic.json"),
    ("saved_chkpt/output_dic.pkl", "saved_chkpt/output_dic.json"),
]

for pkl_path, json_path in FILES:
    # 反序列化回 Dictionary 对象,取其中的 word2index 纯映射
    with open(pkl_path, "rb") as f:
        dic = pickle.load(f)
    with open(json_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保留原始字符;indent=1 换行缩进便于阅读
        json.dump(dic.word2index, f, ensure_ascii=False, indent=1)
    print(f"{pkl_path} -> {json_path}  词表大小: {dic.n_count}")
