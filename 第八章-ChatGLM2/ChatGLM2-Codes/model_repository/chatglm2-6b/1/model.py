# /usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import triton_python_backend_utils as pb_utils
import logging
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import gc
import base64

gc.collect()
torch.cuda.empty_cache()

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class TritonPythonModel:

    def __init__(self):

        self.model_config = None
        self.output_response_dtype = None
        self.output_history_dtype = None
        self.tokenizer = None
        self.model = None

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        output_response_config = pb_utils.get_output_config_by_name(self.model_config, "response")
        output_history_config = pb_utils.get_output_config_by_name(self.model_config, "history")

        self.output_response_dtype = pb_utils.triton_string_to_numpy(output_response_config['data_type'])
        self.output_history_dtype = pb_utils.triton_string_to_numpy(output_history_config['data_type'])

        ChatGLM_path = os.path.dirname(os.path.abspath(__file__)) + "/ChatGLM2_6B"
        checkpoints = os.path.dirname(os.path.abspath(__file__)) + "/checkpoint-300"

        # add by myself
        config = AutoConfig.from_pretrained(ChatGLM_path, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(ChatGLM_path, config=config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(ChatGLM_path, trust_remote_code=True)

        # self.tokenizer = AutoTokenizer.from_pretrained(ChatGLM_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained(ChatGLM_path, torch_dtype=torch.bfloat16, trust_remote_code=True).half().cuda()

        ###
        prefix_state_dict = torch.load(os.path.join(checkpoints, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float()
        ###

        self.model = model.eval()
        logging.info("model init success")

    def execute(self, requests):
        output_response_dtype = self.output_response_dtype
        output_history_dtype = self.output_history_dtype
        responses = []

        logging.info('Request log!')
        print("Request print!")
        logging.info(requests)
        print(requests)
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            prompt = prompt.decode('utf-8')
            print("prompt after:", prompt)
            base64_bytes = prompt.encode('utf-8')
            message_bytes = base64.b64decode(base64_bytes)
            message = str(message_bytes, encoding='utf-8')
            print("message:", message)
            response, history = self.model.chat(self.tokenizer, message, history=[])
            # response, history = self.model.chat(self.tokenizer, prompt, history=[])

            out_log_info = {
                "out_response": response,
                "out_history": history
            }
            logging.info(out_log_info)

            response = np.array(response)
            history = np.array(history)

            response_output_tensor = pb_utils.Tensor("response", response.astype(output_response_dtype))
            history_output_tensor = pb_utils.Tensor("history", history.astype(output_history_dtype))
            final_inference_response = pb_utils.InferenceResponse(output_tensors=[response_output_tensor, history_output_tensor])
            responses.append(final_inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
