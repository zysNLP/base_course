#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:12:55 2021

    Ref:https://github.com/649453932/Chinese-Text-Classification-Pytorch

@author: sunday
"""

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000          # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'     # 未知字，padding符号

def build_vocab(file_path, tokenizer, max_vocab_size, min_freq=1):
    """读取(构造)Embedding字典
    :param file_path: 训练/验证/测试数据路径
    :param tokenizer: 分割词的方式，True为使用空格分割句子，False直接按字分割
    :param max_vocab_size: 词表长度限制, 默认10000
    :param min_freq: 构造词典的最小单词频率, 取值越大，词典单词数越少，使用<UNK>表示的单词越多
    """
    # file_path = config.path_train; max_vocab_size=MAX_VOCAB_SIZE; min_freq=1
    vocab_dict = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # 1. 统计句子中的单词频率
        for line in tqdm(f):
            data = line.strip()
            if not data:
                continue
            content = data.split('\t')[0]
            # break
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        
        # 2. 构造词频数组
        vocab_list = [vocab_value for vocab_value in vocab_dict.items() if vocab_value[1] >= min_freq]
        vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)
        vocab_list = vocab_list[:max_vocab_size]
        
        # 3. 构造Embedding词典
        vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
    
    return vocab_dict


def load_dataset(path, tokenizer, vocab_dict, max_seq_len=32):
    """加载数据集，保存为列表
    :param path: 训练/验证/测试数据路径
    :param tokenizer: 分割词的方式，True为使用空格分割句子，False直接按字分割
    :param vocab_dict: 数据中的单词词典
    :param max_seq_len: 最大序列长度
    """
    # path = config.path_train; vocab_dict=vocab_dict; max_seq_len=32
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            # break
            token = tokenizer(content)
            seq_len = len(token) # pad前的长度(超过max_seq_len的设为max_seq_len)
            if max_seq_len:
                if seq_len < max_seq_len:
                    token.extend([PAD] * (max_seq_len - len(token)))  # 过短padding
                else:
                    token = token[:max_seq_len]                       # 过长截断
                    seq_len = max_seq_len
            # word embedding
            word2idx = []
            for word in token:
                word2idx.append(vocab_dict.get(word, vocab_dict.get(UNK)))
            # contents[句子Embedding, label Embedding， 序列的真实长度]
            contents.append((word2idx, int(label), seq_len))
            
    return contents  


def build_dataset(config, use_word=False):
    """为数据集构造Word Embedding
    :param config: 配置文件对象
    :param use_word: 句子中分隔词/字的方式
    """
    # 1. 定义tokenize的方式
    if use_word:
        tokenizer = lambda x: x.split(' ')    # 以空格隔开，word-level, 适合英文
    else:
        tokenizer = lambda x: [y for y in x]  # 以字符串空隔开，char-level, 适合中文
    # 2. 读取(构造)Embedding字典
    if os.path.exists(config.path_vocab):
        vocab_dict = pkl.load(open(config.path_vocab, 'rb'))
    else:
        vocab_dict = build_vocab(config.path_train, tokenizer=tokenizer, max_vocab_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab_dict, open(config.path_vocab, 'wb'))
    # vocab_dict = build_vocab(config.path_train, tokenizer=tokenizer, max_vocab_size=MAX_VOCAB_SIZE, min_freq=1)
    
    print(f"Vocab size: {len(vocab_dict)}")

    train = load_dataset(config.path_train, tokenizer, vocab_dict, config.max_seq_len)
    dev   = load_dataset(config.path_dev,   tokenizer, vocab_dict, config.max_seq_len)
    test  = load_dataset(config.path_test,  tokenizer, vocab_dict, config.max_seq_len)
    
    return vocab_dict, train, dev, test


class DatasetIterater(object):
    """构造训练/验证/测试数据集的迭代器"""
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        
        # 记录batch的个数
        self.n_batches = len(dataset) // batch_size
        # 记录batch数量是否为整数
        self.residue = True if len(dataset) % self.n_batches != 0 else False
        # 定义一个索引指示器，记录迭batch往前迭代的个数
        self.index = 0
        # 定义一个长度为len(self.dataset)的全0字典，作为__getitem__函数获取下标的字典
        self.count = {}.fromkeys(range(len(self.dataset)), 0)

    def _to_tensor(self, dataset):
        """将数据转为torch.LongTensor类型"""
        x = torch.LongTensor([data[0] for data in dataset]).to(self.device)
        y = torch.LongTensor([data[1] for data in dataset]).to(self.device)
        seq_len = torch.LongTensor([data[2] for data in dataset]).to(self.device)
        return (x, seq_len), y
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """定义迭代器往前迭代的方式"""
        # 1.如果batch数量为整数，同时index等于batch的个数
        if self.residue and self.index == self.n_batches:
            # 定义迭代器中的dataset为传入dataset的第index*batch_size到最后
            dataset = self.dataset[self.index*self.batch_size:]
            dataset = self._to_tensor(dataset)
            self.index += 1
            return dataset
        
        # 2.如果index大于等于batch个数，index赋值为0，抛出迭代异常
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        # 3.其他情况下，定义迭代器中的dataset为传入dataset的第index*batch_size到(index+1)*batch_size
        else:
            dataset = self.dataset[self.index*self.batch_size:(self.index + 1)*self.batch_size]
            dataset = self._to_tensor(dataset)
            self.index += 1
            return dataset

    def __getitem__(self, key):
        """使迭代器具有获取下标的功能"""
        self.count[key] += 1
        return self.dataset[key]

    def __len__(self):
        """使迭代器具有len()函数的功能"""
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
