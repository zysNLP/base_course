#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:09:04 2020

@author: sunday
"""

import numpy as np

"""np.sum()"""
a = np.array([[0, 1, 5],
              [1, 2, 0]])

print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
# array([1, 5])

"""np.dot; np.outer()"""
a = np.array([[0,2,4]])
b = np.array([[2,3,4,5]])
print(a)
print(b)
print(np.dot(a.T,b)) # a.T.shape=(3, 1); b.shape=(1, 4)
print(np.outer(a,b)) # a.shape = (1, 3); b.shape=(1, 4)

"""np.substract()"""
a = np.array([0, 1, 2, 7])
b = np.array([3, 3, 3, 3])
print(a)
print(b)
print(np.subtract(a, b))


