#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:34:52 2021

@author: sunday
"""

# 1. iter, next
string = "Hello"
it = iter(string)
next(it)

# 2. __iter__, __next__
class Fibs:
    def __init__(self, n=10):
        self.a = 0
        self.b = 1
        self.n = n
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > self.n:
            raise StopIteration
        return self.a
    
fibs = Fibs()
next(fibs)
for each in fibs:
    print(each)

from collections import Iterator
isinstance([1, 2, 3], Iterator) 

from collections import Iterable
isinstance([1, 2, 3], Iterable)  

