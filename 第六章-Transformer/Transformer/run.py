#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:57:12 2021

    Ref:https://github.com/649453932/Chinese-Text-Classification-Pytorch

@author: sunday
"""

import torch
import pickle
import numpy as np
from train_eval import train
from model import Config, Model
from utils import build_dataset, build_iterator

def pickle_dump(vocab_dict, train_data, dev_data, test_data, path_data):
    """持久化保存为pickle文件"""
    data_tuple = (vocab_dict, train_data, dev_data, test_data)
    with open(path_data, "wb") as f:
        pickle.dump(data_tuple, f)

def pickle_load(path_data):
    """从pickle文件中读取数据"""
    with open(path_data, "rb") as f:
        data_tuple = pickle.load(f)
    return data_tuple

if __name__ == '__main__':

    # 1.定义配置项
    dataset = './THUCNews'
    embedding = 'random'
    # embedding = 'embedding_SougouNews.npz'
    model_name = 'Transformer'
    config = Config(dataset, embedding, model_name)
    # config_dict = config.__dict__
    # print(config_dict) 
    
    # 2.初始化随机数种子，保证每次运行结果的一致性
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

     # 3.加载数据
    print("Loading data...")                  
    # vocab_dict, train_data, dev_data, test_data = build_dataset(config, False)
    path_data = './THUCNews/data/data.pkl'
    # pickle_dump(vocab_dict, train_data, dev_data, test_data, path_data)
    vocab_dict, train_data, dev_data, test_data = pickle_load(path_data)

    train_iter = build_iterator(train_data, config)
    dev_iter   = build_iterator(dev_data,   config)
    test_iter  = build_iterator(test_data,  config)
    
    # train_data0 = train_data[0]
    # train_iter0 = train_iter[0]
    # assert train_data[0] == train_iter[0]
    
    # 4.模型训练
    config.n_vocab = len(vocab_dict)
    model = Model(config).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
