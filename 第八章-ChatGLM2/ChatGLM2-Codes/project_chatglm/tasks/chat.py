# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：chat.py.py
    @Author  ：ys
    @Time    ：2023/11/16 14:36 
"""

from transformers import AutoTokenizer, AutoModel, AutoConfig


class Chat:
    def __init__(self):
        # 加载模型
        self.model_path = "/HDD2/pretrain_model/chatglm2-6b"

    def init_model(self):
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(self.model_path, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return tokenizer, model

