# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：chatglm.py
    @Author  ：ys
    @Time    ：2023/11/10 18:55 
"""

from flask import request, jsonify
from loguru import logger

from . import app
from tasks.main import Test
from tasks.chat import Chat
from instance.config import configs
from tasks.define import define_model

chat = Chat()
logger.info("开始加载模型...")
tokenizer, model = chat.init_model()
# model = model.eval()
logger.info("模型加载完成!")

checkpoints = "/HDD2/zys/codes/po-master/ChatGLM2-6B/ptuning/output/po/po-chatglm2-6b-extract-Lenze-128-2e-2/checkpoint-300"
model = define_model(checkpoints, model)
logger.info("Checkpoint加载完成!")


@app.route('/chatglm/write_homework', methods=['POST'])
def write_homework():
    try:
        data = request.json  # 获取 JSON 格式的请求数据
        text = data.get("text")  # 从请求数据中获取需要处理的文本
        result, history = model.chat(tokenizer, text, history=[])
        # t = Test()
        # result = t.run(text, **configs)
        logger.info(f"预测结果：{result = }")
        return jsonify({"result": result})  # 返回处理结果
    except BaseException as e:
        logger.error(e)
        return e