# 微调后
import copy
import os
import torch
# import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "/HDD2/pretrain_model/chatglm2-6b"

path = "./ChatGLM2-6B/ptuning/data/po/780S/predis.json"
with open(path, "r", encoding='utf-8') as f:
    preds = f.readlines()
path = "./ChatGLM2-6B/ptuning/data/po/780S/valids.json"
with open(path, "r", encoding='utf-8') as f:
    valid = f.readlines()
preds = [eval(pred) for pred in preds]
valid = [eval(pred) for pred in valid]

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

checkpoints = "/HDD2/zys/codes/po-master/ChatGLM2-6B/ptuning/output/po/po-chatglm2-6b-extract-780S-128-2e-2/checkpoint-300"
prefix_state_dict = torch.load(os.path.join(checkpoints, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

# np.save("./prompt", prefix_state_dict["transformer.prefix_encoder.embedding.weight"].to('cpu').numpy())

model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

results = []
for c in range(len(preds)):
    content = preds[c]["content"]
    summary = preds[c]["summary"]

    response, history = model.chat(tokenizer, content, history=[], top_p=0.1, temperature=01.0, do_sample=False)
    print(f"summary：{summary}")
    print(f"response：{response}")
    print("----")

print("success")
