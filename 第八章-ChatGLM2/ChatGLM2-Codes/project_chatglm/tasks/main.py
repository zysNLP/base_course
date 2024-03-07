# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：main.py
    @Author  ：ys
    @Time    ：2023/11/10 19:01 
"""

from tasks.chat import Chat
from loguru import logger


class Test:

    def __init__(self):
        self.chat = Chat()

    def run(self, content, **kwargs):
        logger.info("开始加载模型...")
        tokenizer, model = self.chat.init_model()
        logger.info("模型加载完成!")
        response, history = model.chat(tokenizer, content, history=[])
        logger.info(f"预测结果：{response = }")
        return response


if __name__ == "__main__":

    t = Test()
    res = t.run("abc")
    print(res)