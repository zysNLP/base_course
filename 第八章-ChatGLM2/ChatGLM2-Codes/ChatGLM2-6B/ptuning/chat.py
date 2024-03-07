# -*- coding: utf-8 -*- 
"""
    @Project ：ptuning 
    @File    ：chat.py
    @Author  ：ys
    @Time    ：2023/10/9 18:45 
"""

# 加载模型
model_path = "./models"
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

content = "你好，你是谁？"

response, history = model.chat(tokenizer, content, history=[], top_p=0.1, temperature=01.0, do_sample=False)

print(f"response：{response}")

