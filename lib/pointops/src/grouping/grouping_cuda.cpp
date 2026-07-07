#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "grouping_cuda_kernel.h"


void grouping_forward_cuda(int m, int nsample, int c, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor output_tensor)
{
    const float *input = input_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *output = output_tensor.data<float>();
    grouping_forward_cuda_launcher(m, nsample, c, input, idx, output);
}

void grouping_backward_cuda(int m, int nsample, int c, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor grad_input_tensor)
{
    const float *grad_output = grad_output_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_input = grad_input_tensor.data<float>();
    grouping_backward_cuda_launcher(m, nsample, c, grad_output, idx, grad_input);
}
