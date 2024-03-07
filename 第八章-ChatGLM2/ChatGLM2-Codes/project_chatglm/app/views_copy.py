# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：views_copy.py.py
    @Author  ：ys
    @Time    ：2023/11/16 15:10 
"""

from flask import request, jsonify
from loguru import logger

from . import app
from tasks.main import Test
from instance.config import configs


@app.route('/chatglm/write_homework', methods=['POST'])
def write_homework():
    try:
        data = request.json  # 获取 JSON 格式的请求数据
        text = data.get("text")  # 从请求数据中获取需要处理的文本
        t = Test()
        result = t.run(text, **configs)
        return jsonify({"result": result})  # 返回处理结果
    except BaseException as e:
        logger.error(e)
        return e