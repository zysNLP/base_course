#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:15:24 2020

@author: sunday
"""



import numpy as np
from lstm import LSTM
from utils import gen_data

def main():
    
    # 1. 构造数据
    # X[0]=[[0, 1, 1, 1, 1], [1, 0, 0, 1, 1]]; y_int2[0]=[0, 1, 0, 0, 0, 1, 0]
    X, y_int2, y_int10 = gen_data(20, 2, 5) # (sample_nums, input_dims, feature_lens)
    
    # A sample of data is [X[0], Y[0]]
    X = np.array(X)        # X.shape = (sample_nums, input_dims, feature_lens)
    Y = np.array(y_int10)  # Y.shape = (sample_nums, input_dims)

    # 2. 类实例化
    lstm = LSTM(5, 2)  # input_size, hidden_size
    
    # 3. 模型训练
    learning_rate = 1e-3
    for epoch in range(3000):
        for i in range(len(X)):
            
            # 前向传播
            y_pred = lstm.forward(X[i])  #array([[0.43069228, 0.        ]])
            # 损失函数
            loss = np.square(y_pred - Y[i]).sum()
            # 梯度计算
            grad_y_pred = 2.0 * (y_pred - Y[i])  #array([[-67.13861544, -68.        ]])
            # 反向传播
            lstm.backward(grad_y_pred)
            # 参数更新
            lstm.update_weight(learning_rate)
        
        # 每1000步打印一次
        if epoch == 999 or epoch % 10 == 0:
            total_loss = 0
            for i in range(len(X)):
                y_pred = lstm.forward(X[i])
                loss = np.square(y_pred - Y[i]).sum()
                total_loss += loss
            print("epoch {} loss {:.6f}".format(epoch, total_loss / len(X)))
            print("epoch {} loss {:.6f}".format(epoch, total_loss / len(X)))

    # 4. 测试
    print('\nstart test...')
    print('=' * 50)
    X, y_int2, y_int10 = gen_data(5, 2, 5)
    X = np.array(X)
    Y = np.array(y_int10)
    for i in range(len(X)):
        y_pred = lstm.forward(X[i])
        print('X:\n', X[i])
        print('true: {}  predict: {:.2f}\n'.format(Y[i][0], y_pred[0][0]))

    print('\n测试长度为3的序列:')
    print('-' * 50)
    X, y_int2, y_int10 = gen_data(5, 3, 5)
    X = np.array(X)
    Y = np.array(y_int10)
    for i in range(len(X)):
        y_pred = lstm.forward(X[i])
        print('X:\n', X[i])
        print('true: {}  predict: {:.2f}\n'.format(Y[i][0], y_pred[0][0]))



if __name__ == '__main__':
    main()


