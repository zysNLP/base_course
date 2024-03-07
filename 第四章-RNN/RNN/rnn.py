# coding=utf-8
# Ref: https://github.com/hf136/models
from __future__ import print_function
from __future__ import division

import numpy as np

__author__ = 'wuyueqiu'


class RNNCell:
    """
    一个 RNN 时间步的计算过程
    """
    def __init__(self, in_size, hidden_size):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.w_i2h = np.random.normal(0, 0.1, (in_size, hidden_size))
        self.w_h2h = np.random.normal(0, 0.1, (hidden_size, hidden_size))
        self.bias = np.random.normal(0, 0.1, (1, hidden_size))

    def relu(self, x): #array([[ 0.30320593, -0.13429247]])
        x[x < 0] = 0
        return x

    def forward(self, i, h):         # array([0, 1, 1, 1, 1]), array([0., 0.])
        self.i2h = i.dot(self.w_i2h) # array([ 0.19844807, -0.23888629])
        self.h2h = h.dot(self.w_h2h) # array([0., 0.])
        self.h_relu = self.relu(self.i2h + self.h2h + self.bias)
        return self.h_relu

    def backward(self, grad, i, h): #array([[-67.13861544, -68.]]); array([1, 0, 0, 1, 1]); array([[0.30320593, 0. ]])
        if i.ndim == 1:
            i = np.expand_dims(i, axis=0)
        if h.ndim == 1:
            h = np.expand_dims(h, axis=0)

        self.grad_h_relu = grad
        self.grad_h = self.grad_h_relu.copy()
        self.grad_h[h < 0] = 0
        self.grad_w_h2h = h.T.dot(self.grad_h)
        self.grad_w_i2h = i.T.dot(self.grad_h)
        self.grad_bias = self.grad_h
        self.grad_h_in = self.grad_h.dot(self.w_h2h.T)

        return self.grad_h_in

    def update_weight(self, lr):
        self.w_i2h -= lr * self.grad_w_i2h
        self.w_h2h -= lr * self.grad_w_h2h
        self.bias -= lr * self.grad_bias


class RNN:
    """
    完整的 RNN 序列计算过程
    """
    def __init__(self, in_size, hidden_size):         # (5, 2)
        self.in_size = in_size          # 5
        self.hidden_size = hidden_size  # 2
        self.rnncell = RNNCell(in_size, hidden_size)  # (5, 2)

    def forward(self, x): # array([[0, 1, 1, 1, 1],[1, 0, 0, 1, 1]])
        self.h_state = []
        self.x = x
        h = np.zeros(self.hidden_size)
        for i in x:
            self.h_state.append(h)
            h = self.rnncell.forward(i, h) # h每次被更新后保存到h_state
        self.h_out = h
        return self.h_out

    def backward(self, grad): #array([[-67.13861544, -68.        ]])
        self.grad_w_i2h = np.zeros((self.in_size, self.hidden_size))
        self.grad_w_h2h = np.zeros((self.hidden_size, self.hidden_size))
        self.grad_bias = np.zeros((1, self.hidden_size))

        for i in range(len(self.h_state) - 1, -1, -1): #[1, 0]
            x = self.x[i]
            h = self.h_state[i]
            grad = self.rnncell.backward(grad, x, h)
            self.grad_w_i2h += self.rnncell.grad_w_i2h
            self.grad_w_h2h += self.rnncell.grad_w_h2h
            self.grad_bias += self.rnncell.grad_bias
        return grad

    def update_weight(self, lr):
        self.rnncell.w_i2h -= lr * self.grad_w_i2h
        self.rnncell.w_h2h -= lr * self.grad_w_h2h
        self.rnncell.bias -= lr * self.grad_bias
        return self.rnncell.w_i2h, self.rnncell.w_h2h, self.rnncell.bias
