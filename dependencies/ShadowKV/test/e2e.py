################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

from data.dataset import Dataset
os.chdir(root_dir)

from models import choose_model_class

dataset_name = "ruler/qa_2"

configs = {
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "01-ai/Yi-9B-200K": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 10,
            "shadowkv_bsz": 42,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 5,
            "shadowkv_bsz": 21,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 10,
        }
    },
    "THUDM/glm-4-9b-chat-1m": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 12,
            "shadowkv_bsz": 50,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 6,
            "shadowkv_bsz": 25,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 3,
            "shadowkv_bsz": 12,
        }
    }
}


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "meta-llama/Meta-Llama-3.1-8B-Instruct", "01-ai/Yi-9B-200K","THUDM/glm-4-9b-chat-1m"])
    p.add_argument("--datalen", type=str, default="122k", choices=["60k", "122k", "244k"])

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()

    model_name = args.model_name
    length = args.datalen

    min_prompt_len = configs[model_name][length]["min_prompt_len"]
    temperature = 0.6
    baseline_bsz = configs[model_name][length]["baseline_bsz"]
    shadowkv_bsz = configs[model_name][length]["shadowkv_bsz"]
    sparse_budget = configs[model_name][length]["sparse_budget"]


    ##################### Baseline #####################
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_name, device='cuda:0',  batch_size=baseline_bsz, max_length=min_prompt_len, attn_mode='full', sparse_budget=sparse_budget)
    dataset = Dataset(dataset_name, llm.tokenizer, 256*1024, 20)

    input_ids = torch.cat([dataset[i][0][:, :min_prompt_len] for i in range(llm.batch_size)], dim=0)

    assert input_ids.shape[-1] == min_prompt_len

    _, throughput_baseline = llm.batch_generate(input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    print(colored(f"[Baseline] Throughput: {throughput_baseline} tokens/s", 'red'))

    del llm.kv_cache
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ##################### ShadowKV #####################

    llm = LLM(model_name=model_name, device='cuda:0',  batch_size=shadowkv_bsz, max_length=min_prompt_len, attn_mode='shadowkv_cpu', sparse_budget=sparse_budget)
    dataset = Dataset(dataset_name, llm.tokenizer, 256*1024, 100)

    input_ids = torch.cat([dataset[i][0][:, :min_prompt_len] for i in range(llm.batch_size)], dim=0)
    _, throughput_shadowkv = llm.batch_generate(input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    print(colored(f"[ShadowKV] Throughput: {throughput_shadowkv} tokens/s", 'red'))
    
    print(colored(f"Speedup: {throughput_shadowkv / throughput_baseline:.2f}x", 'red'))