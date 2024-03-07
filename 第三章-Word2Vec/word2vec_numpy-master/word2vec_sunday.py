#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:40:40 2020

    Word2Vec代码原理详解
    Ref: https://github.com/DerekChia/word2vec_numpy

@author: sunday
"""

import os
import numpy as np
from collections import defaultdict

# 1.读取数据
def read_data():
    text1 = "natural language processing and Machine learning is fun and exciting"
    text2 = "natural language processing and Python is nice and exciting"
    
    texts = [text1, text2]
    corpus = [[word.lower() for word in text.split()] for text in texts]
    
    return corpus

corpus = read_data()

# 2.构造Embedding
word_counts = defaultdict(int)
for row in corpus:
    for word in row:
       word_counts[word] += 1

v_count = len(word_counts.keys())
words_set = list(word_counts.keys())

word_index = dict((word, i) for i, word in enumerate(words_set))
index_word = dict((i, word) for i, word in enumerate(words_set))

# One-Hot函数
def word2onehot(word, v_count, word_index):
    # word = 'and'
    word_vec = [0 for i in range(0, v_count)]
    word_index = word_index[word]
    word_vec[word_index] = 1
    return word_vec

# word = words_set[3]
# word_vec = word2onehot(word, v_count, word_index)
# print(len(word_vec))

# 3.构造训练数据
def get_training_data(corpus, v_count, word_index):
    
    training_data = []
    for sentence in corpus:
        # sentence = corpus[0]
        sent_len = len(sentence)
        for i, word in enumerate(sentence):
            # i = 3; word = 'and'
            w_target  = word2onehot(sentence[i], v_count, word_index)
            w_contexts = []
            for j in range(i - 2, i + 2 + 1):
                if j >= 0 and j != i and j < (sent_len-1):
                    # print(j)
                    w_context = word2onehot(sentence[j], v_count, word_index)
                    w_contexts.append(w_context)
                    
            training_data.append([w_target, w_contexts])
           
    return np.array(training_data)

training_data = get_training_data(corpus, v_count, word_index)

# softmax
def softmax(u):
    e_u = np.exp(u - np.max(u))
    return e_u / e_u.sum(axis=0)

# 前向传播
def forward_prop(w_t, w1, w2):
    h = np.dot(w_t, w1) # 10
    u = np.dot(h, w2)   # 9
    yp = softmax(u)     # 9
    return yp, h, u

# 后向传播
def backward_prop(e, h, w_t, w1, w2, lr):
    dLdw2 = np.outer(h, e)  # (N_h, N_e)-->(10, 9)
    dLdw1 = np.outer(w_t, np.dot(w2, e.T)) # (9, (10,9)*(9,1))-->(9, 10)
    w1 = w1 - (lr * dLdw1)  # (9, 10) - (1*(9, 10))
    w2 = w2 - (lr * dLdw2)  # (10, 9) - (1*(10, 9))
    return  w1, w2

lr = 0.01
epochs = 100
np.random.seed(1234)
w1 = np.random.uniform(-1, 1, (len(index_word), 100)) # print(w1.shape)
w2 = np.random.uniform(-1, 1, (100, len(index_word))) # print(w2.shape)

# 4.训练
def train(training_data, w1, w2, lr, epochs):
    
    for epoch in range(epochs):
        # epoch = 0
        loss = 0
        for w_t, w_c in training_data:
            # w_t, w_c = training_data[3]; print(len(w_t))
            yp, h, u = forward_prop(w_t, w1, w2)
            # wc = w_c[0]
            wc_sub = np.array([np.subtract(yp, wc) for wc in w_c])
            e = np.sum(wc_sub, axis=0)
            
            w1, w2 = backward_prop(e, h, w_t, w1, w2, lr)
            
            loss += -np.sum([u[wc.index(1)] for wc in w_c]) + \
                len(w_c) * np.log(np.sum(np.exp(u)))
    
        print("Epoch:", epoch, "Loss:", loss)

train(training_data, w1, w2, lr, epochs)

# 获取词向量
def word_vec(word, word_index, w1):
    w_index = word_index[word]
    v_w = w1[w_index]
    return  v_w

# 5.计算相似度
def vec_sim(word, v_count, top_n, w1, word_index, index_word):
    
    # 获取当前词的词向量
    v_w1 = word_vec(word, word_index, w1)
    
    # 计算获取当前词与所有词的相似度集合
    word_sim = {}
    for i in range(v_count):
        v_w2 = w1[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den

        word = index_word[i]
        word_sim[word] = theta
    
    # 相似度结果排序
    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
    for word, sim in words_sorted[:top_n]:
        print(word, sim)

word = "language"
vec = word_vec(word, word_index, w1)
vec_sim(word, v_count, 5, w1, word_index, index_word)
















