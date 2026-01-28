# ArkVale: Efficient Gener<ins>a</ins>tive LLM Inference with <ins>R</ins>ecallable <ins>K</ins>ey-<ins>Val</ins>ue <ins>E</ins>viction 

[\[Link\]](https://neurips.cc/virtual/2024/poster/96635) [\[Paper\]](./media/arkvale-nips24-paper.pdf) [\[Poster\]](./media/arkvale-nips24-poster.pdf) [\[Slides\]](./media/arkvale-nips24-talk.pdf)


## BoruiXu changes

- Modify some codes to support Transformers library 4.47
- Due to the original C++ kernels does not support $\frac{Attention_{head}}{KeyVale_{head}} \neq [1,4,8]$, I replicate cache to assume the key_value_head equals attention_head.


Comment: 
- The code uses full budget pages for attention computation, and ensures the top-k pages in the budget pool.
- The result quality will be impacted when running the code on a GPU that is running multi-tasks.

## Download

```bash
git clone https://github.com/pku-liang/ArkVale.git --recursive 
```

or 

```bash
git clone https://github.com/pku-liang/ArkVale.git
cd ArkVale
git submodule update --init --recursive --depth 1 
```

## Install 

```bash
pip install -r requirements.txt
cd source && python3 setup.py [develop|install]
```

## Usage 

```python
from transformers import AutoModelForCausalLM
from arkvale import adapter
path = ...
dev = torch.device("cuda:0")
dtype = torch.float16
model = (
    AutoModelForCausalLM
    .from_pretrained(path, torch_dtype=dtype, device_map=dev)
    .eval()
)
adapter.enable_arkvale(
    model, 
    dtype=dtype, 
    device=dev, 
    page_size=32,
    # page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
    page_budgets=4096 // 32,
    page_topks=32,
    n_max_bytes=40 * (1 << 30),
    n_max_cpu_bytes=80 * (1 << 30),
)
...
```
