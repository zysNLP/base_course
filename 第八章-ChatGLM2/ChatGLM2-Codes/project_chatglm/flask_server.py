# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：flask_server.py
    @Author  ：ys
    @Time    ：2023/11/10 19:11 
"""

from app import app
# 导入视图函数，使得app.route生效
from app.views import *

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12666)


