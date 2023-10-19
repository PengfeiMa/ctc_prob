#include <ctc_forward_prob.h>
#include <custom_ops.h>

#include <ATen/Tensor.h>
// #include <torch/extension.h>
#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnl_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/util/tensor_util.h"
#include "aten/util/types.h"

using namespace torch_mlu;

void ctc_forward_prob_mlu(torch::Tensor r, torch::Tensor log_phi, torch::Tensor x_, int start, int end) {

  auto r_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(r);
  auto r_impl = getMluTensorImpl(r_contiguous);
  auto r_ptr = r_impl->cnnlMalloc();

  auto s_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(log_phi);
  auto s_impl = getMluTensorImpl(s_contiguous);
  auto s_ptr = s_impl->cnnlMalloc();
  
  // auto x_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(x_);
  auto x_permute = torch_mlu::cnnl::ops::cnnl_permute(x_, at::IntArrayRef{1, 0, 2, 3});
  auto x_permute_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(x_permute);
  auto x_impl = getMluTensorImpl(x_permute_contiguous);
  auto x_ptr = x_impl->cnnlMalloc();

  size_t num_per_compute = r.size(2) * r.size(3);

  cnrtQueue_t queue = getCurQueue();

  ctc_forward_prob_kernel_entry(queue,
                                reinterpret_cast<float *>(r_ptr),
                                reinterpret_cast<float *>(s_ptr),
                                reinterpret_cast<float *>(x_ptr),
                                reinterpret_cast<float *>(r_ptr),
                                start, end, num_per_compute);

}

PYBIND11_MODULE(torch_mlu_ext_ops, m) {
  m.def("ctc_forward_prob_mlu", &ctc_forward_prob_mlu);
}