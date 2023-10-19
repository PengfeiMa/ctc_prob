#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void ctc_forward_prob_mlu(torch::Tensor r, torch::Tensor log_phi, torch::Tensor x_, int start, int end);