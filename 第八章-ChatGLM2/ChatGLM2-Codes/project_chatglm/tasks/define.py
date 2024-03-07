# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：define.py
    @Author  ：ys
    @Time    ：2023/11/17 09:51 
"""

import os
import torch


def define_model(checkpoints, model):
    prefix_state_dict = torch.load(os.path.join(checkpoints, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    return model