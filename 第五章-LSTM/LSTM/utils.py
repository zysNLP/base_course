#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:45:44 2020

@author: sunday
"""

import numpy as np


def int10_to_int2(num, size=5):
    res = []
    while num > 0:
        res.insert(0, num % 2)
        num = num // 2
    while len(res) < size:
        res.insert(0, 0)
    return res


def gen_data(data_size, seq_len, vec_dim):
    X = []
    y_int2 = []
    y_int10 = []
    for i in range(data_size):
        int2 = []
        int10 = []
        for j in range(seq_len):
            rand = np.random.randint(0, 2 ** vec_dim)
            int10.append(rand)
            int2.append(int10_to_int2(rand, vec_dim))
        X.append(int2)
        t = sum(int10)
        y_int10.append([t])
        y_int2.append(int10_to_int2(t, vec_dim + 2))
    
    return X, y_int2, y_int10




