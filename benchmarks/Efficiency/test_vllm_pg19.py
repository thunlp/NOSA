import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.cuda import nvtx
from vllm import LLM, SamplingParams

"""
vllm==0.7.2
"""

dataset = load_dataset("emozilla/pg19")['test']['text']
path = "/somepath/Megatron-LM/hf_checkpoints/8b_full_sft_llama/900"
tokenizer = AutoTokenizer.from_pretrained(path)

B = 4
L = 16 * 1024
max_new_tokens = 4
test_n = 4

model = LLM(model=path)



print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")

params = SamplingParams(temperature=0, max_tokens=max_new_tokens+1, ignore_eos=True)

@torch.inference_mode()
def test_time(input_ids):

    input_strs = tokenizer.batch_decode(input_ids)

    output = model.generate(input_strs, params)

    last_fist_token_time = max(output[i].metrics.first_token_time for i in range(B))
    last_last_token_time = max(output[i].metrics.last_token_time for i in range(B))
    decoding_time = last_last_token_time - last_fist_token_time
    thru = B * max_new_tokens / decoding_time
    return thru


total_t = 0
add_time = 0
first_time = True
for i in range(len(dataset)):

    text = dataset[i]

    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda").input_ids
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

