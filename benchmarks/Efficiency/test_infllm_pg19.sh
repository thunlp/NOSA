set -x

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=PASSIVE
export KMP_AFFINITY=granularity=fine,compact,1,0

export TORCH_CUDA_ARCH_LIST=8.0 # Change to your GPU architecture

CUDA_VISIBLE_DEVICES=0 \
taskset -c 0-25 \
python test_infllm_pg19.py

