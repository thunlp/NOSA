#include "pytorch_extension_utils.h"
#include <cutlass/util/device_rmsnorm.h>

torch::Tensor rms_norm(torch::Tensor input,  // [bsz, len, hidden_dim]
                       torch::Tensor weight, // [hidden_dim]
                       float epsilon) {
  int m = input.size(0) * input.size(1);
  int n = input.size(2);

  torch::Tensor output = torch::empty_like(input);

  bool success =
      DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input.scalar_type(), c_type, [&] {
        dim3 grid(m);
        if (n % 8 == 0 && std::is_same<c_type, nv_half>::value) {
          dim3 block(min(1024, (n / 8 + 31) / 32 * 32));

          cutlass::rmsnorm_twoPassAlgo_e8<<<grid, block, 0, nullptr>>>(
              (float4 *)output.data_ptr(), (const float4 *)input.data_ptr(),
              (const float4 *)weight.data_ptr(), m, n, epsilon);
        } else {
          dim3 block(min(1024, ((n + 31) / 32 + 31) / 32 * 32));

          cutlass::rmsnorm_twoPassAlgo_e1<<<grid, block, 0, nullptr>>>(
              (c_type *)output.data_ptr(), (c_type *)input.data_ptr(),
              (c_type *)weight.data_ptr(), m, n, epsilon);
        }

        auto status = cudaGetLastError();
        TORCH_CHECK(status == cudaSuccess, "rms_norm failed with error code ",
                    cudaGetErrorString(status));
        return true;
      });

  return output;
}
