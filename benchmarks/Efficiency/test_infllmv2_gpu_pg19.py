import time
import torch
import gc
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.cuda import nvtx
from nosi import InfLLMv2Llama as Llama


dataset = load_dataset("emozilla/pg19")['test']['text']
path = "/somepath/Megatron-LM/hf_checkpoints/8b_infllmv2_sft_llama/900"
tokenizer = AutoTokenizer.from_pretrained(path)
model = Llama(
    model_name=path,
    device="cuda",
    offload=False
)

B = 4
L = 16 * 1024
max_new_tokens = 4
test_n = 4

print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")



def test_time(input_ids):
    gen_ids, thru = model.batch_generate_benchmark(input_ids, max_new_tokens=max_new_tokens+2)
    # gc.collect()
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

n_tokens = max_new_tokens * B 
tok_per_s = total_t / add_time
decode_time = n_tokens / tok_per_s
print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")