#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knnquery_cuda_kernel.h"


void knnquery_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    const float *xyz = xyz_tensor.data<float>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const int *offset = offset_tensor.data<int>();
    const int *new_offset = new_offset_tensor.data<int>();
    int *idx = idx_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();
    knnquery_cuda_launcher(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}
