#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "sampling_cuda_kernel.h"


void furthestsampling_cuda(int b, int n, at::Tensor xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor tmp_tensor, at::Tensor idx_tensor)
{
    const float *xyz = xyz_tensor.data<float>();
    const int *offset = offset_tensor.data<int>();
    const int *new_offset = new_offset_tensor.data<int>();
    float *tmp = tmp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    furthestsampling_cuda_launcher(b, n, xyz, offset, new_offset, tmp, idx);
}
