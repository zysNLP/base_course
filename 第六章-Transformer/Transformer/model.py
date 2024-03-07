#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:08:41 2021

    Ref:https://github.com/649453932/Chinese-Text-Classification-Pytorch

@author: sunday
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding, model_name):
        
        self.dataset = dataset
        self.embedding = embedding
        self.model_name = model_name
        
        """1.初始化路径相关的变量"""
        self.path_train = self.dataset + '/data/train.txt'                                # 训练集路径
        self.path_dev = self.dataset + '/data/dev.txt'                                    # 验证集路径
        self.path_test = self.dataset + '/data/test.txt'                                  # 测试集路径
        self.path_vocab = self.dataset + '/data/vocab.pkl'                                # 唯一词表路径
        self.path_save = self.dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 训练结果保存路径
        self.path_log = self.dataset + '/log/' + self.model_name                          # 训练log保存路径
        
        """2.初始化Embedding相关的变量"""
        self.embedding_pretrained = self.define_embedding_pretrained()  # 指定是否使用预训练好的词向量模型
        self.embedding_dim = self.define_embedding_dim()                # 指定Embedding的维度
        
        """3.初始化模型训练相关的变量"""
        self.device = self.device()                                     # 指定设备类型
        self.n_vocab = 0                                                # 词表大小，在运行时赋值        
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_seq_len = 32                                           # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率        
        self.dropout = 0.5                                              # Dropout
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.class_list = self.read_class_list()                        # 类别列表
        self.num_classes = len(self.class_list)                         # 类别数
        
        """4.初始化模型结构相关的变量"""
        self.model_dim = 300                                            # Encoder层的输入层维度 
        self.hidden_size = 1024                                         # Encoder层的隐藏层维度 
        self.num_heads = 5                                              # 多头注意力机制的注意力头个数
        self.num_encoders = 2                                           # Encoder层的个数
    
    def read_class_list(self):
        """读取所有分类标签"""
        return [x.strip() for x in open(self.dataset + '/data/class.txt', encoding='utf-8').readlines()]
    
    def define_embedding_pretrained(self):
        """定义是否使用预训练的词向量模型"""
        if self.embedding != 'random':
            return torch.tensor(np.load(self.dataset + '/data/' + self.embedding)["embeddings"].astype('float32'))
        else:
            return None

    def define_embedding_dim(self):
        """定义词向量维度"""
        if self.embedding_pretrained != None:
            return self.embedding_pretrained.size(1)
        else:
            return 300
    
    def device(self):
        """指定设备类型"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Attention Is All You Need'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        # 1.定义Word Embedding层
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=config.n_vocab, 
                                          embedding_dim=config.embedding_dim, 
                                          padding_idx=config.n_vocab-1)
        
        # 2.定义Position Embedding层
        self.postion_embedding = Positional_Encoding(embedding_dim=config.embedding_dim, 
                                                     max_seq_len=config.max_seq_len, 
                                                     dropout=config.dropout, 
                                                     device=config.device)
        
        # 3.定义Encoder层
        self.encoder = Encoder(model_dim=config.model_dim, 
                               num_heads=config.num_heads, 
                               hidden_size=config.hidden_size, 
                               dropout=config.dropout)
        
        # 4.定义多个Encoder层
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoders)])

        # 5.定义全连接层
        self.fc1 = nn.Linear(in_features=config.max_seq_len * config.model_dim, 
                             out_features=config.num_classes)

    def forward(self, x):
        # x[0].shape=(batch_size, max_seq_len)
        out = self.embedding(x[0])         # (batch_size, max_seq_len, embedding_dim)
        out = self.postion_embedding(out)  # (batch_size, max_seq_len, embedding_dim)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    """定义Encoder层"""
    def __init__(self, model_dim, num_heads, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(model_dim, num_heads, dropout)
        self.feed_forward = Position_wise_Feed_Forward(model_dim, hidden_size, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out
    

class Positional_Encoding(nn.Module):
    """定义位置编码层"""
    def __init__(self, embedding_dim, max_seq_len, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        # 根据论文中的公式，构造PE矩阵
        self.position_encoding = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embedding_dim)) \
                                                for i in range(embedding_dim)] for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, max_seq_len, embedding_dim) + (max_seq_len, embedding_dim)
        out = x + nn.Parameter(self.position_encoding, requires_grad=False).to(self.device) # help_tensor.py
        # (batch_size, max_seq_len, embedding_dim)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''定义缩放版的点积注意力层'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        # (1024, 32, 64) * (1024, 64, 32) >> (1024, 32, 32)
        attention = torch.matmul(Q, K.permute(0, 2, 1))    # help_tensor.py
        # attention = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    """定义多头注意力机制层"""
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        
        self.num_heads = num_heads
        assert model_dim % num_heads == 0  # 是否可以被整除，余数为0
        self.dim_head = model_dim // self.num_heads
        self.fc_Q = nn.Linear(model_dim, num_heads * self.dim_head)
        self.fc_K = nn.Linear(model_dim, num_heads * self.dim_head)
        self.fc_V = nn.Linear(model_dim, num_heads * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_heads * self.dim_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        
        batch_size = x.size(0)
        
        Q = self.fc_Q(x)   # (batch_size, max_seq_len, embedding_dim) (128, 32, 300)
        K = self.fc_K(x)   # (batch_size, max_seq_len, embedding_dim) (128, 32, 300)
        V = self.fc_V(x)   # (batch_size, max_seq_len, embedding_dim) (128, 32, 300)
        
        # (batch_size*num_heads, max_seq_len, dim_head)
        Q = Q.view(batch_size * self.num_heads, -1, self.dim_head) # (640, 32, 60)
        K = K.view(batch_size * self.num_heads, -1, self.dim_head) # (640, 32, 60)
        V = V.view(batch_size * self.num_heads, -1, self.dim_head) # (640, 32, 60)
        
        # Q = Q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2) # (128, 5, 32, 60)
        # K = K.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2) # (128, 5, 32, 60)
        # V = V.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2) # (128, 5, 32, 60)
        
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_heads, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)
        
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head)
        context = context.view(batch_size, -1, self.dim_head * self.num_heads)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    """"""
    def __init__(self, model_dim, hidden_size, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # out = self.layer_norm(self.dropout(self.fc2(F.relu(self.fc1(x)))) + x)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out










