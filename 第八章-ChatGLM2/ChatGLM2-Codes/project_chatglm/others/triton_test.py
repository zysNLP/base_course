# -*- coding: utf-8 -*- 
"""
    @Project ：project_chatglm 
    @File    ：triton_test.py
    @Author  ：ys
    @Time    ：2023/11/20 19:16 
"""

from transformers import BertTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import time

# 1、准备词嵌入后的数据
# 加载预训练的 tokenizer
tokenizer = BertTokenizer.from_pretrained('/data/NFS/MODELS/stella-large-zh-v2')
# tokenizer = BertTokenizer.from_pretrained('/data/NFS/MODELS/bge-large-zh')

# 准备您的文本数据
text = "可以看出人的性别吗？"

# 编码文本
encoded_inputs = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    return_tensors="pt",
    max_length=1024
)

# 获取encoded的数据
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
token_type_ids = encoded_inputs['token_type_ids']

# 如果您需要将这些张量转换为NumPy数组，可以调用.numpy()，例如：
input_ids_np = input_ids.numpy()
attention_mask_np = attention_mask.numpy()
token_type_ids_np = token_type_ids.numpy()

# 2、校验服务
# 现在 input_ids_np, attention_mask_np, token_type_ids_np 可以被用于Triton推理请求
# Triton服务的URL，假设是在本地运行
triton_url = '10.65.250.175:8030'

# 创建一个Triton客户端实例

client = httpclient.InferenceServerClient(url=triton_url)

# 检查Triton服务是否已启动
if not client.is_server_live():
    print("Triton Server isn't live")
if not client.is_server_ready():
    print("Triton Server isn't ready")

# 检查模型是否已加载并准备好推理
model_name = 'stella_large_zh_v2'
if not client.is_model_ready(model_name):
    print(f"Model {model_name} isn't ready")

# 3、使用 Triton 服务推理
# 准备输入数据
input_ids = input_ids_np.astype('int32')
attention_mask = attention_mask_np.astype('int32')
token_type_ids = token_type_ids_np.astype('int32')

# 创建输入数据的字典
inputs = [
    httpclient.InferInput('input_ids', [1, 1024], "INT32"),
    httpclient.InferInput('attention_mask', [1, 1024], "INT32"),
    httpclient.InferInput('token_type_ids', [1, 1024], "INT32"),
]

inputs[0].set_data_from_numpy(input_ids, binary_data=False)
inputs[1].set_data_from_numpy(attention_mask, binary_data=False)
inputs[2].set_data_from_numpy(token_type_ids, binary_data=False)

# 定义输出
outputs = [
    httpclient.InferRequestedOutput('output_0', binary_data=False),
    httpclient.InferRequestedOutput('output_1', binary_data=False),
]
# 发送推理请求
start_time = time.time()
results = client.infer(model_name, inputs, outputs=outputs)
end_time = time.time()

print(end_time - start_time)

# 获取输出结果
output0_data = results.as_numpy('output_0')
output1_data = results.as_numpy('output_1')

# 打印结果
# print("output_0 data:", output0_data)
print("output_1 data:", output1_data)