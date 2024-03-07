# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：test.py
    @Author  ：ys
    @Time    ：2023/11/10 19:13 
"""


import requests
import json

data = {"text": "Terms of delivery (Incoterms 2020) FOB , FOB Exc. packaging & insurance {'Item': '010', 'Material': '13626018', 'Req.date of arrival': '22.07.2022', 'Quant.': '30 Piece', 'Value in': '2,455.60'} O-description ACC Manufacturer Shenzhen Jove Enterprise Co., Ltd"}
# data = {"text": "你好"}  # 修改为 "text" 键

post_json = json.dumps(data, ensure_ascii=False).encode("utf-8")
headers = {'Content-Type': 'application/json'}  # 设置请求头为 JSON 格式
result = requests.post("http://127.0.0.1:12666/chatglm/write_homework", data=post_json, headers=headers)
result = result.json()
print(result)