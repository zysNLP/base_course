# -*- coding: utf-8 -*-

import json
import base64
import requests
import numpy as np
from typing import List

import warnings

warnings.filterwarnings("ignore")


class ChatGLM2:
    max_token = 32000
    temperature = 0.01
    history_len = 1
    url = 'http://localhost:8000/v2/models/chatglm2-6b/infer'

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
                {"name": "prompt", "datatype": "BYTES", "shape": [1], "data": [query]},
                {"name": "history", "datatype": "BYTES", "shape": [1], "data": history},
                {"name": "temperature", "datatype": "BYTES", "shape": [1], "data": [temperature]},
                {"name": "max_token", "datatype": "BYTES", "shape": [1], "data": [max_token]},
                {"name": "history_len", "datatype": "BYTES", "shape": [1], "data": [history_len]}
            ],
            "outputs": [{"name": "response"},
                        {"name": "history"}]
        }
        response = requests.post(url=url,
                                 data=json.dumps(data, ensure_ascii=True),
                                 headers={"Content_Type": "application/json"},
                                 timeout=120)
        result = json.loads(response.text)
        print(result)

        return result

    def chat(self,
             query: str,
             history: List[List[str]] = []):
        temperature = str(self.temperature)
        max_token = str(self.max_token)
        history_len = str(self.history_len)
        url = self.url
        response = self._infer(url, query, history, temperature, max_token, history_len)
        print(response)


if __name__ == "__main__":
    glm2 = ChatGLM2()
    glm2.chat(query="hello", history=['hi'])