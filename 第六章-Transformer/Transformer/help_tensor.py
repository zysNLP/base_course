#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:25:18 2021

@author: sunday
"""

"""--------------Tensor运算 + """
import torch

# 这两个Tensor加减乘除会对b自动进行Broadcasting
a = torch.rand(3, 4)
b = torch.rand(4)

c1 = a + b
c2 = torch.add(a, b)
print(c1.shape, c2.shape)
print(torch.all(torch.eq(c1, c2)))

"""--------------Tensor运算 matmul """
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()
# torch.Size([])

# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# torch.Size([3])

# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3])

# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3, 5])

# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3, 5])


import torch.nn.functional as F

a=torch.rand(3,4,5)
b=F.softmax(a,dim=0) # b[0][0][0] + b[1][0][0] + b[2][0][0]
c=F.softmax(a,dim=1) # c[0][0][0] + c[0][1][0] + c[0][2][0] + c[0][3][0]
d=F.softmax(a,dim=2) # d[0][0][0] + d[0][0][1] + d[0][0][2] + d[0][0][3] + d[0][0][4]
torch.eq(F.softmax(a,dim=2), F.softmax(a,dim=-1))















