# coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
from rnn import RNN


__author__ = 'wuyueqiu'


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


def main():
    X, y_int2, y_int10 = gen_data(20, 2, 5)
    X = np.array(X)
    Y = np.array(y_int10)
    rnn = RNN(5, 1)

    learning_rate = 1e-3
    for epoch in range(1000):
        for i in range(len(X)):
            
            y_pred = rnn.forward(X[i])

            # 损失函数
            loss = np.square(y_pred - Y[i]).sum()

            # 计算梯度
            grad_y_pred = 2.0 * (y_pred - Y[i])
            rnn.backward(grad_y_pred)
            rnn.update_weight(learning_rate)

        if epoch == 999 or epoch % 10 == 0:
            total_loss = 0
            for i in range(len(X)):
                y_pred = rnn.forward(X[i])
                loss = np.square(y_pred - Y[i]).sum()
                total_loss += loss
            print("epoch {} loss {:.6f}".format(epoch, total_loss / len(X)))

    # 测试效果
    print('\nstart test...')
    print('=' * 50)
    X, y_int2, y_int10 = gen_data(5, 2, 5)
    X = np.array(X)
    Y = np.array(y_int10)
    for i in range(len(X)):
        y_pred = rnn.forward(X[i])
        print('X:\n', X[i])
        print('true: {}  predict: {:.2f}\n'.format(Y[i][0], y_pred[0][0]))

    print('\n测试长度为3的序列:')
    print('-' * 50)
    X, y_int2, y_int10 = gen_data(5, 3, 5)
    X = np.array(X)
    Y = np.array(y_int10)
    for i in range(len(X)):
        y_pred = rnn.forward(X[i])
        print('X:\n', X[i])
        print('true: {}  predict: {:.2f}\n'.format(Y[i][0], y_pred[0][0]))

    # 打印 RNN 模型权重信息
    print('\nRNN weights:')
    print('-' * 50)
    print(rnn.rnncell.w_i2h)
    print(rnn.rnncell.w_h2h)
    print(rnn.rnncell.bias)


if __name__ == '__main__':
    main()
