from models import Llama
from transformers import LlamaTokenizer

llama = Llama(
    model_name='/home/test/test01/hyx/Megatron-LM/hf_ckpts_1b_new_data/1000',
    sparse_budget=4096,
    attn_mode='shadowkv',
)
tokenizer = LlamaTokenizer.from_pretrained('/home/test/test01/hyx/Megatron-LM/hf_ckpts_1b_new_data/1000')

input_str = "The grass is green. The sky is blue. The flower is red. " * 300 + "Now neglect the previous input. Which city is the capital of China? Answer:"
input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(llama.device)
breakpoint()
output = llama.generate(input_ids, gen_len=16, temperature=0)

breakpoint()