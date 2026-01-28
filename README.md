<div align="center">

<h1>NOSA: Native and Offloadable Sparse Attention</h1>


**Boost Decoding Efficiency via High-Locality Offloading**
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/thunlp/NOSA" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="TODO" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/NOSA-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2510.13602" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-2510.13602-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huangyuxiang03.github.io/blogs_nosa" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/Blog-000000?style=for-the-badge&logo=googlechrome&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Overview

**NOSA** is a trainable sparse attention mechanism designed for KV-cache offloading with an explicit locality constraint, paired with an inference system (**NOSI**) to realize its efficiency. It improves long-context/long-generation quality over prior offloading baselines while boosting decoding throughput by up to **5.04×** vs **FullAttn**, **1.92×** vs **InfLLMv2**, and **1.83×** vs **ShadowKV** on **1B/3B/8B** LLMs.

<img width="4189" height="1200" alt="framework_github" src="https://github.com/user-attachments/assets/0e8d51af-4257-40f4-94bb-263856d23ee3" />



## Models

We train 1B, 3B, and 8B models  FullAttn, InfLLMv2, DMA, and NOSA, resulting in a total of 12 models. The following models have been released on Hugging Face.

|Model|Link|
|:-:|:-:|
|NOSA-1B | [NOSA-1B](huggingface.co/openbmb/NOSA-1B) |
|NOSA-3B | [NOSA-3B](huggingface.co/openbmb/NOSA-3B) |
|NOSA-8B | [NOSA-8B](huggingface.co/openbmb/NOSA-8B) |

Please reach out to us if additional baseline models (FullAttn, InfLLMv2, or DMA) are needed. You may open an issue or contact us directly via email (our email addresses are provided in the paper).





## Setup

We set up our experimental environment using uv inside Docker. If you need to set up the docker from scratch, please refer to `dependencies/README.md`

First, download our Docker image from ModelScope: [huangyx21/nosa-env-docker](https://modelscope.cn/models/huangyx21/nosa-env-docker/).
Please start from this image, where most dependencies have been pre-installed to greatly simplify environment setup.
```
pip install modelscope
modelscope download --model huangyx21/nosa-env-docker nosa_comp.tar --local_dir ./dependencies
docker import ./dependencies/nosa_comp.tar nosa:newest
# Then, please set up the directory mapping in ./dependencies/launch_docker.sh manually.
bash ./dependencies/launch_docker.sh nosa:newest # This takes a while
```

We have pre-installed four virtual environments. You can activate each one by executing the corresponding command below. For vLLM and SGLang evaluations, please refer to their official Docker images.
```
source /venv/nosa/bin/activate # for NOSA, FullAttn, InfLLMv2, DMA
source /venv/shadowkv/bin/activate # for ShadowKV
source /venv/infllm/bin/activate # for InfLLM
source /venv/arkvale/bin/activate # for arkvale
```

For flexibility, we keep the evaluation framework LM-Harness-Eval for general tasks, and ShadowKV not pre-installed. Please install these two packages as follows. Assume we are in the repo's directory now.
- LM-Harness-Eval:
```
source /venv/nosa/bin/activate
uv pip install -e benchmarks/lm-evaluation-harness
```
- ShadowKV:
```
source /venv/shadowkv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.0 # Change to your GPU architecture
cd dependencies/ShadowKV
uv pip install -e . --no-build-isolation
```

Also, please install NOSI as follows.
```
uv pip install ./nosi
```



## Run Experiments

### Long-Input Evaluation

We run all methods on LongBench and HELMET.

- LongBench
```
cd benchmarks/LongBench

# download test data
bash download_data.sh
# activate the corresponding venv
source /venv/nosa/bin/activate
# run LongBench
python pred.py --model 8b_nosa_sft
python eval.py --model 8b_nosa_sft

cd -
```

- HELMET
```
cd benchmarks/HELMET

# download test data
bash scripts/download_data.sh
# activate the corresponding venv
source /venv/nosa/bin/activate
# run HELMET
python eval.py --output_dir output
bash collect_result.sh

cd -

```

### General Tasks

```
cd benchmarks/lm-evaluation-harness

# activate the corresponding venv
source /venv/nosa/bin/activate
bash run_nosa.sh && bash run_infllmv2.sh && bash run_full.sh && bash run_dma.sh

cd -
```

### Decoding Efficiency Tests

Each setting has a `test_xxx_pg19.sh` in `benchmarks/Efficiency`. Directly running them can obtain the decoding throughput.

```
cd benchmarks/Efficiency

# activate the corresponding venv
source /venv/nosa/bin/activate
# for example: NOSA+NOSI
bash test_nosa_pg19.sh

cd -
```

## Acknowledgment

Some content of this repository are adapted from [LongBench](https://github.com/THUDM/LongBench), [HELMET](https://github.com/princeton-nlp/HELMET), [RULER](https://github.com/NVIDIA/RULER), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [ShadowKV](https://github.com/ByteDance-Seed/ShadowKV), [ArkVale](https://github.com/pku-liang/ArkVale/), [InfLLM](https://github.com/thunlp/InfLLM), and [InfLLMv2](http://github.com/OpenBMB/infllmv2_cuda_impl/).

## Citation

```
@article{huang2025nosa,
  title={NOSA: Native and Offloadable Sparse Attention},
  author={Huang, Yuxiang and Wang, Pengjie and Han, Jicheng and Zhao, Weilin and Su, Zhou and Sun, Ao and Lyu, Hongya and Zhao, Hengyu and Wang, Yudong and Xiao, Chaojun and Han, Xu and Liu, Zhiyuan},
  journal={arXiv preprint arXiv:2510.13602},
  year={2025}
}
```
