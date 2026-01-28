import time
import torch
from transformers import AutoTokenizer
# from proxy_modeling_infllmv2 import SparseLlamaForCausalLM
from proxy_modeling_nosa import SparseLlamaForCausalLM  # 你的模型类

# ==== 加载模型 ====
# path = "/somepath/Megatron-LM/hf_ckpts_1b_new_data_infllmv2_bugfix_sft/300"
path = "/somepath/Megatron-LM/hf_ckpts_1b_new_data_nosa_bugfix_move_exp_sft/300"
tokenizer = AutoTokenizer.from_pretrained(path)
model = SparseLlamaForCausalLM.from_pretrained(
    path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
model.eval()

# ==== 构造输入 ====
input_str = (
    "<|im_start|>user\n"
    + "What is the meaning of life? " * 2000
    + "Now, neglect previous context and answer the following question. "
    +  "Give a brief introduction of Tsinghua University. <|im_end|>\n<|im_start|>assistant\n"
)
input_ids = tokenizer(input_str, return_tensors="pt").to("cuda").input_ids.repeat(64, 1)
B, L = input_ids.shape
max_new_tokens = 6

print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")



def test_time():

    # ==== Prefill ====
    with torch.no_grad():
        print("[Prefill] running ...")
        prefill_out = model(input_ids, use_cache=True)
        past_key_values = prefill_out.past_key_values
        next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.empty_cache()
    # ==== Decode (显式 greedy 循环) ====
    new_tokens = [next_token]
    with torch.no_grad():
        # 第一个 decode step（第二个 token）——仅 warmup，不计时
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        new_tokens.append(next_token)

        # ====== 从这里开始计时 (第三个 token 起) ======
        torch.cuda.synchronize()
        t_start = time.time()
        # breakpoint()
        for step in range(2, max_new_tokens):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)
            past_key_values = outputs.past_key_values
            # breakpoint()

        torch.cuda.synchronize()
        t_end = time.time()
    return t_end - t_start

test_time()

total_t = 0
for _ in range(10):
    t = test_time()
    print(f"time: {t}")
    total_t += t


# # ==== 拼接输出 ====
# output = torch.cat([input_ids] + new_tokens, dim=1)
# decoded = tokenizer.decode(output[0])
# print(" ..." + decoded[-400:])

# ==== 性能统计 ====
decode_time = total_t / 10
n_tokens = (max_new_tokens - 2) * B  # 从第3个 token 开始计时
tok_per_s = n_tokens / decode_time
print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")
a = 1
