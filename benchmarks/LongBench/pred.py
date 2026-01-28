import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from inf_llm.utils import GreedySearch
except:
    print("no infllm in environment")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, 
                        choices=[
                            "1b_nosa_sft",
                            "3b_nosa_sft",
                            "8b_nosa_sft",
                            "1b_fullattn_sft",
                            "3b_fullattn_sft",
                            "8b_fullattn_sft",
                            "1b_infllmv2_sft",
                            "3b_infllmv2_sft",
                            "8b_infllmv2_sft",
                            "1b_infllmv1_64",
                            "3b_infllmv1_64",
                            "8b_infllmv1_64",
                            "1b_infllmv1_128",
                            "3b_infllmv1_128",
                            "8b_infllmv1_128",
                            "1b_shadowkv_chunk_size=8",
                            "3b_shadowkv_chunk_size=8",
                            "8b_shadowkv_chunk_size=8",
                            "1b_shadowkv_chunk_size=64",
                            "3b_shadowkv_chunk_size=64",
                            "8b_shadowkv_chunk_size=64",
                            "1b_arkvale_32",
                            "3b_arkvale_32",
                            "8b_arkvale_32",
                            "8b_nosa_pref",
                            "8b_shadowkv_minf_chunk_size=8",
                            "8b_shadowkv_minf_chunk_size=64",
                            "8b_dma",
                            "8b_dma_pref",
                            "1b_dma",
                            "1b_dma_pref",
                            "3b_dma",
                            "3b_dma_pref",
                            "1b_nosa_pref",
                            "3b_nosa_pref"
                            "1b_shadowkv_minf_chunk_size=8",
                            "1b_shadowkv_minf_chunk_size=64",
                            "3b_shadowkv_minf_chunk_size=8",
                            "3b_shadowkv_minf_chunk_size=64",
                            "8b_nosa_sft_sb=8",
                            "8b_nosa_sft_sb=16",
                            "8b_nosa_sft_sb=24",
                            "8b_nosa_sft_sb=32",
                        ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "minicpm" in model_name:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

@torch.no_grad()
def gen(model, input_ids, max_new_tokens, eos_token_ids):
    output = model(input_ids, use_cache=True)
    generated_ids = []
    next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_id.item())
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, past_key_values=past_key_values)
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
        past_key_values = output.past_key_values
        if next_id.item() in eos_token_ids:
            break
    return generated_ids

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if "shadowkv" in model_name:
            q_len = input.input_ids.shape[-1]
            if q_len <= 4*1024:
                model.set_mode("full")
            else:
                model.set_mode("shadowkv")
            pred = model.generate(input.input_ids, gen_len=max_gen, temperature=0, top_k=-1, top_p=1.0)[0]
        elif "arkvale" in model_name:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                use_cache=False,
            )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        elif "infllmv1" in model_name:
            q_len = input.input_ids.shape[-1]
            searcher = GreedySearch(model, tokenizer)

            pred = searcher.generate(input_ids=input.input_ids, max_length=max_gen)
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
            )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "nosa" in model_name or "infllmv2" in model_name or "fullattn" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "shadowkv" in model_name:
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        import sys
        sys.path.insert(0, "/home/test/test01/hyx/ShadowKV")
        from models import Llama
        
        if "minf" in model_name:
            print("use minference")
        
        model = Llama(
            model_name=path,
            sparse_budget=3072, # local + outliers = 1024
            attn_mode='shadowkv',
            rank=40,
            device=str(device),
            chunk_size=64,
            minference=True if "minf" in model_name else False,
        )
    elif "arkvale" in model_name:
        try:
            from arkvale import adapter
        except:
            raise ValueError("no arkvale in environment")

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
        adapter.enable_arkvale(
            model,
            dtype=torch.float16,
            device=device,
            page_size=32,
            page_budgets=128,
            page_topks=65, # k + nw + ns - 1
            n_sink_pages=2,
            n_win_pages=32,
            n_max_bytes=40 * (1 << 30),
            n_max_cpu_bytes=60 * (1 << 30),
        )
    elif "infllmv1" in model_name:
        try:
            from inf_llm.utils import patch_hf, GreedySearch
            from omegaconf import OmegaConf
        except:
            raise ValueError("no infllm in environment")
        
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16).to(device)
        inf_config = OmegaConf.load("./model_configs/infllmv1/infllmv1-128.yaml" if "128" in model_name else "./model_configs/infllmv1/infllmv1-64.yaml").model
        model = patch_hf(model, inf_config.type, **inf_config)
    elif "llama2" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    else:
        raise ValueError("Unknown")
    
    # model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    # world_size = 1
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "gov_report",
            "triviaqa",
            "narrativeqa",
            "qmsum",
            "musique",   
            "2wikimqa",
            "multifieldqa_en",
            "repobench-p",
            "hotpotqa",
            "trec",
            "passage_retrieval_en",
            "passage_count",
            "samsum"
        ]
    else:
        datasets = [
            "gov_report",
            "triviaqa",
            "narrativeqa",
            "qmsum",
            "musique",   
            "2wikimqa",
            "multifieldqa_en",
            "repobench-p",
            "hotpotqa",
            "trec",
            "passage_retrieval_en",
            "passage_count",
            "samsum"
        ]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # predict on each dataset
    if not args.e and not os.path.exists("pred"):
        os.makedirs("pred")
    if args.e and not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    tokenizer = AutoTokenizer.from_pretrained(model2path[model_name])
    cnt = []
    
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('json', data_files=f"./data/{dataset}.jsonl")['train']
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"

        data = data.map(lambda x: {"combined_text": x["context"] + x["input"]})
        
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
