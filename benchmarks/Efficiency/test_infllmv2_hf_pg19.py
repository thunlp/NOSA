import time
import torch
from transformers import AutoTokenizer
from proxy_modeling_infllmv2 import SparseLlamaForCausalLM
from datasets import load_dataset
from torch.cuda import nvtx
import gc


dataset = load_dataset("emozilla/pg19")['test']['text']
path = "/somepath/Megatron-LM/hf_checkpoints/8b_infllmv2_sft_llama/900"
tokenizer = AutoTokenizer.from_pretrained(path)
model = SparseLlamaForCausalLM.from_pretrained(
    path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
model.eval()


B = 4
L = 16 * 1024
max_new_tokens = 4
test_n = 4

print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")

def test_time(input_ids):

    model.timer_beg = 0
    model.timer_end = 0
    model.passed_iters = 0
    output = model.generate(input_ids, max_new_tokens=max_new_tokens + 2, do_sample=False)
    model.timer_end = time.time()
    gc.collect()
    return model.timer_end - model.timer_beg


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
    print(f"time: {t}")
    if first_time:
        first_time = False
    else:
        total_t += t
        add_time += 1
    if add_time == test_n:
        break


decode_time = total_t / test_n
n_tokens = (max_new_tokens) * B 
tok_per_s = n_tokens / decode_time
print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")

