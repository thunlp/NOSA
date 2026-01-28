# Setting up Environment from Scratch

Here are the instructions of setting up the environment from scratch. 
We use uv, venv, and docker to setup our experiment environment.

## Starting from NGC Docker

We start our experiment from the docker image `nvcr.io/nvidia/pytorch:25.01-py3` by 
```
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run \
  -v <your local path>:<your in-docker path> \
  -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --network host \
  --privileged=true \
  --entrypoint bash \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:25.01-py3
```

Then, setup uv.
```
apt-get update
apt-get install uv
```

## Making Virtual Environments

Next, we setup four virtual environments for NOSA, ShadowKV, InfLLM, and ArkVale. For FullAttn, InfLLMv2, and DMA, we use NOSA's environment. For vLLM and SGLang, please refer to their official dockers.

- NOSA's environment (also for FullAttn, InfLLMv2, and DMA)

```
uv venv /venv/nosa --python 3.10
source /venv/nosa/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv pip install nemo-toolkit[all] # for ablations on RULER, optional
uv pip install -r requirements_nosa.txt 
FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install flash_attn==2.6.3 --no-build-isolation --no-cache
uv pip install flashinfer-python==0.5.3 --no-build-isolation
uv pip install ./infllmv2_cuda_impl --no-build-isolation
uv pip install ./flash-attention-nosa --no-build-isolation
```

- ShadowKV

```
uv venv /venv/shadowkv --python 3.10
source /venv/shadowkv/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv pip install -r requirements_shadowkv.txt 
# minference
uv pip install minference==0.1.5 --no-build-isolation
# vllm, flash_attn
uv pip install vllm==0.5.3.post1
uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.3/
uv pip install torch==2.3.0
FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install flash_attn==2.8.3 --no-build-isolation --no-cache
# MiniCPM4's tokenizer needs tokenizer>=0.22.0
# So we fix the dependency table by a in-place patch
cat dependency_versions_table.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/dependency_versions_table.py
uv pip install tokenizers==0.22.0
# Also, fix modeling_llama.py to support newer models
cat modeling_llama_shadowkv.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
```

- InfLLM

```
uv venv /venv/infllm --python 3.10
source /venv/infllm/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv pip install -r requirements_infllm.txt 
# MiniCPM4's tokenizer needs tokenizer>=0.22.0
# So we fix the dependency table by a in-place patch
cat dependency_versions_table.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/dependency_versions_table.py
uv pip install tokenizers==0.22.0
# install infllm
uv pip install ./InfLLM
# Also, fix modeling_llama.py to support newer models
cat modeling_llama_infllm.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
```

- ArkVale

```
uv venv /venv/arkvale --python 3.10
source /venv/arkvale/bin/activate
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv pip install -r requirements_arkvale.txt 
# MiniCPM4's tokenizer needs tokenizer>=0.22.0
# So we fix the dependency table by a in-place patch
cat dependency_versions_table.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/dependency_versions_table.py
uv pip install tokenizers==0.22.0
# Also, fix modeling_llama.py to support newer models
cat modeling_llama_arkvale.py > /venv/shadowkv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
```

Then, compile ArkVale. Please make sure you have a good internet connection, and please refer to [ArkVale](https://github.com/pku-liang/ArkVale) for possible compilation problems.
```
cd ArkVale/source
python setup_all.py install
```
If everything goes well, you can run the following command successfully:
```
python -c "import arkvale"
```

## Finishing Environmental Setup

After the above process, you are able to obtain an docker image with 4 virtual environments. Please activate the corresponding venv for each experiment settings.

```
source /venv/nosa/bin/activate # FullAttn, InfLLMv2, DMA, NOSA
source /venv/shadowkv/bin/activate # ShadowKV
source /venv/infllm/bin/activate # InfLLM
source /venv/arkvale/bin/activate # ArkVale
```