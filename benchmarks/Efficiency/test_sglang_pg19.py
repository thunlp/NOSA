import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.cuda import nvtx
import sglang as sgl

"""
sglang==0.4.7
"""

if __name__ == "__main__":

    path = "/somepath/Megatron-LM/hf_checkpoints/8b_full_sft_llama/900"
    model = sgl.Engine(model_path=path, disable_radix_cache=True)

    dataset = load_dataset("emozilla/pg19")['test']['text']
    tokenizer = AutoTokenizer.from_pretrained(path)

    B = 4
    L = 64 * 1024
    max_new_tokens = 4
    test_n = 4




    print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")


    @torch.inference_mode()
    def test_time(input_ids, max_new_tokens):

        input_strs = tokenizer.batch_decode(input_ids)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        params = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        }
        torch.cuda.synchronize()
        start.record()
        output = model.generate(input_strs, params)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000

        return elapsed_time


    total_t = 0
    add_time = 0
    first_time = True
    test_data = []
    for i in range(len(dataset)):

        text = dataset[i]

        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda").input_ids
        if input_ids.shape[1] < L:
            continue
        input_ids = input_ids[:, :L].repeat(B, 1)
        test_data.append(input_ids)
        if first_time:
            first_time = False
        else:
            add_time += 1
        if add_time == test_n:
            break

    print(f"prefill: {test_time(test_data[0], 1)}")
    prefill_t = 0
    for td in test_data[1:]:
        pt = test_time(input_ids, 1)
        prefill_t += pt
        print(f"prefill: {pt}")


    print(f"all: {test_time(test_data[0], max_new_tokens+1)}")
    all_t = 0
    for td in test_data[1:]:
        allt = test_time(input_ids, max_new_tokens+1)
        all_t += allt
        print(f"all: {allt}")
    
    decode_time = all_t - prefill_t
    n_tokens = (max_new_tokens) * B  * test_n
    tok_per_s = n_tokens / decode_time
    print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
    print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")

