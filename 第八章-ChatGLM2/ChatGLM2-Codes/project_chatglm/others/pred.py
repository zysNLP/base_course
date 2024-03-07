# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：pred.py
    @Author  ：ys
    @Time    ：2023/11/21 12:19 
"""

import json
import time
import base64
import struct
import requests
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from langchain.llms.base import LLM
from langchain.llms import OpenAI
from langchain.llms.utils import enforce_stop_tokens
from typing import Dict, List, Optional, Tuple, Union, Mapping, Any

import warnings

warnings.filterwarnings("ignore")


class ChatGLM(LLM):
    max_token = 32000
    temperature = 0.01
    history_len = 10
    url = "http://localhost:8000/v2/models/chatglm2-6b/infer"

    def __init__(self):
        super(ChatGLM, self).__init__()

    @property
    def _llm_type(self):
        return "ChatGLM2-6B"

    @property
    def _history_len(self) -> int:
        return self.history_len

    @property
    def _max_token(self) -> int:
        return self.max_token

    @property
    def _temperature(self) -> float:
        return self.temperature

    def _deserialize_bytes_tensor(self, encoded_tensor):
        """
        Deserializes an encoded bytes tensor into an
        numpy array of dtype of python objects
        Parameters
        ----------
        encoded_tensor : bytes
            The encoded bytes tensor where each element
            has its length in first 4 bytes followed by
            the content
        Returns
        -------
        string_tensor : np.array
            The 1-D numpy array of type object containing the
            deserialized bytes in 'C' order.
        """
        strs = list()
        offset = 0
        val_buf = encoded_tensor
        while offset < len(val_buf):
            l = struct.unpack_from("<I", val_buf, offset)[0]
            offset += 4
            sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
            offset += l
            strs.append(sb)
        return (np.array(strs, dtype=np.object_))

    @classmethod
    def _infer(cls, url, query, history, temperature, max_token, history_len):
        query = base64.b64encode(query.encode('utf-8')).decode('utf-8')
        history_origin = np.asarray(history).reshape((-1,))
        history = [base64.b64encode(item.encode('utf-8')).decode('utf-8') for item in history_origin]
        temperature = base64.b64encode(temperature.encode('utf-8')).decode('utf-8')
        max_token = base64.b64encode(max_token.encode('utf-8')).decode('utf-8')
        history_len = base64.b64encode(history_len.encode('utf-8')).decode('utf-8')
        data = {
            "model_name": "chatglm2-6b",
            "inputs": [
                {"name": "prompt", "datatype": "BYTES", "shape": [1], "contents": {"bytes_contents": [query]}},
                {"name": "history", "datatype": "BYTES", "shape": [-1], "contents": {"bytes_contents": history}},
                {"name": "temperature", "datatype": "BYTES", "shape": [1],
                 "contents": {"bytes_contents": [temperature]}},
                {"name": "max_token", "datatype": "BYTES", "shape": [1], "contents": {"bytes_contents": [max_token]}},
                {"name": "history_len", "datatype": "BYTES", "shape": [1],
                 "contents": {"bytes_contents": [history_len]}}
            ],
            "outputs": [{"name": "response"},
                        {"name": "history"}]
        }
        response = requests.post(url=url,
                                 data=json.dumps(data, ensure_ascii=True),
                                 headers={"Content_Type": "application/json"},
                                 timeout=120)
        return response

    def _call(self,
              query: str,
              history: List[List[str]] = [],
              stop: Optional[List[str]] = None):
        temperature = str(self.temperature)
        max_token = str(self.max_token)
        history_len = str(self.history_len)
        url = self.url
        response = self._infer(url, query, history, temperature, max_token, history_len)
        if response.status_code != 200:
            return "查询结果错误"
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        result = json.loads(response.text)
        # 处理response
        res = base64.b64decode(result['raw_output_contents'][0].encode('utf-8'))
        res_response = self._deserialize_bytes_tensor(res)[0].decode()
        return res_response

    def chat(self,
             query: str,
             history: List[List[str]] = [],
             stop: Optional[List[str]] = None):
        temperature = str(self.temperature)
        max_token = str(self.max_token)
        history_len = str(self.history_len)
        url = self.url
        response = self._infer(url, query, history, temperature, max_token, history_len)
        if response.status_code != 200:
            return "查询结果错误"
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        result = json.loads(response.text)
        # 处理response
        res = base64.b64decode(result['raw_output_contents'][0].encode('utf-8'))
        res_response = self._deserialize_bytes_tensor(res)[0].decode()
        # 处理history
        history_shape = result['outputs'][1]["shape"]
        history_enc = base64.b64decode(result['raw_output_contents'][1].encode('utf-8'))
        res_history = np.array([i.decode() for i in self._deserialize_bytes_tensor(history_enc)]).reshape(
            history_shape).tolist()
        return res_response, res_history

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }
        return _param_dict

