#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_M 64

extern "C" __global__ void diff_kernel(
    const int64_t* __restrict__ old_map,   // [H,B,M]
    const int64_t* __restrict__ new_act,   // [H,B,M]
    int64_t* __restrict__ new_map,         // [H,B,M]
    int64_t* __restrict__ load_map,        // [H,B,M]
    int H, int B, int M
){
    int h  = blockIdx.x;
    int b  = blockIdx.y;
    int m  = threadIdx.x;   // slot index

    if (m >= M) return;

    int base = (h * B + b) * M;

    extern __shared__ int64_t smem[];
    int64_t* sm_old_hit = smem;
    int64_t* sm_new_hit = smem + M;

    sm_old_hit[m] = 0;
    sm_new_hit[m] = 0;

    int64_t old_val = old_map[base + m];
    int64_t new_val = new_act[base + m];

    // 先看old_map[h, b, m]是否存在于new_act
    bool old_hit = false;
    for (int64_t j = 0; j < M; ++j) {
        if (old_val == new_act[base + j]) {
            old_hit = true;
            break;
        }
    }

    if (old_hit) {
        sm_old_hit[m] = 1; // 在sm上标记old[h, b, m]命中
        new_map[base + m] = old_val;  // 这个位置的映射关系不变
        load_map[base + m] = -1; // 不从cpu load
    }
    
    // 然后看new_map[h, b, m]是否存在于old中
    bool new_hit = false;
    for (int64_t j = 0; j < M; ++j) {
        if (new_val == old_map[base + j]) {
            new_hit = true;
            break;
        }
    }

    if (new_hit) {
        sm_new_hit[m] = 1;
    } 

    __syncthreads(); // 同步一下保证sm上的old和new hit状态都写完

    // 从 new 没有hit入手，计算是第几个没有hit的
    int64_t k = -1;
    if (sm_new_hit[m] == 0) {
        int64_t cnt = 0;
        for (int64_t j = 0; j < m; j++) {
            if (sm_new_hit[j] == 0) {
                cnt++;
            }
        }
        k = cnt;
    }
    
    // 找到放new[h, b, m]的位置
    int64_t target_slot = -1;
    if (sm_new_hit[m] == 0) {
        int64_t cnt = 0;
        for (int64_t j = 0; j < M; j++) {
            if (sm_old_hit[j] == 0) {
                if (cnt == k) {
                    target_slot = j;
                    break;
                }
                cnt++;
            }
        }
    }

    if (target_slot >= 0) {
        int64_t write_idx = base + target_slot;
        int64_t new_val_m = new_act[base + m];

        new_map[write_idx] = new_val_m;
        load_map[write_idx] = new_val_m;
    }
}




void diff_offload_cuda(
    at::Tensor old_map,     // [H,B,M] in
    at::Tensor new_act,     // [H,B,M] in 
    at::Tensor new_map,     // [H,B,M] out
    at::Tensor load_map     // [H,B,M] out
){
    int H = old_map.size(0);
    int B = old_map.size(1);
    int M = old_map.size(2);

    dim3 grid(H, B);


    dim3 block(M);

    diff_kernel<<<grid, block, sizeof(int64_t) * 2 * M>>>(
        old_map.data_ptr<int64_t>(),
        new_act.data_ptr<int64_t>(),
        new_map.data_ptr<int64_t>(),
        load_map.data_ptr<int64_t>(),
        H, B, M
    );
}
