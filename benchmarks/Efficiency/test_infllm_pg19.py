import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.cuda import nvtx
from inf_llm.utils import patch_hf, GreedySearch
from omegaconf import OmegaConf


dataset = load_dataset("emozilla/pg19")['test']['text']
path = "/somepath/Megatron-LM/hf_checkpoints/8b_full_sft_llama/900"
tokenizer = AutoTokenizer.from_pretrained(path)

B = 16
L = 16 * 1024
max_new_tokens = 4
test_n = 4

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")


inf_config = OmegaConf.load("./infllmv1-128.yaml").model
model = patch_hf(model, inf_config.type, **inf_config)
searcher = GreedySearch(model, tokenizer)


print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")



@torch.inference_mode()
def test_time(input_ids):

    pred, decode_time = searcher.generate(input_ids=input_ids, max_length=max_new_tokens, benchmark=True)

    thru = B * max_new_tokens / decode_time

    return thru


total_t = 0
add_time = 0
first_time = True
for i in range(len(dataset)):

    text = dataset[i]

    input_ids = tokenizer(text, return_tensors="pt").to("cuda").input_ids
    if input_ids.shape[1] < L:
        continue
    input_ids = input_ids[:, :L].repeat(B, 1)
    
    t = test_time(input_ids)
    print(f"thru: {t} tok/s")
    if first_time:
        first_time = False
    else:
        total_t += t
        add_time += 1
    if add_time == test_n:
        break

n_tokens = (max_new_tokens) * B 
tok_per_s = total_t / add_time
decode_time = n_tokens / tok_per_s
print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")

