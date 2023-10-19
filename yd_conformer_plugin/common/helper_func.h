/*************************************************************************
 *  * Copyright (C) [2020-2023] by Cambricon, Inc.
 *   *************************************************************************/
#ifndef SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_HELPER_FUNC_H_
#define SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_HELPER_FUNC_H_
#include <vector>
#include <string>

float Diff1(float *ptr_mlu, float *ptr_cpu, int len) {
  float diff_sum = 0;
  float sum      = 0;
  for (int i = 0; i < len; i++) {
    float diff = std::fabs(ptr_cpu[i] - ptr_mlu[i]);
    diff_sum += diff;
    sum += std::fabs(ptr_cpu[i]);
  }
  sum += 1e-9;
  return diff_sum / sum;
}

float Diff2(float *ptr_mlu, float *ptr_cpu, int len) {
  float diff_sum = 0;
  float sum      = 0;
  for (int i = 0; i < len; i++) {
    float diff = (ptr_cpu[i] - ptr_mlu[i]) * (ptr_cpu[i] - ptr_mlu[i]);
    diff_sum += diff;
    sum += (ptr_cpu[i] * ptr_cpu[i]);
  }
  sum += 1e-9;
  return std::sqrt(diff_sum / sum);
}

class inputManager {
 public:
  explicit inputManager(int byte_size) : byte_size(byte_size) {
    CNRT_CHECK(cnrtMalloc(&mlu_input_ptr, byte_size));
  }
  ~inputManager() {
    if (mlu_input_ptr) {
      CNRT_CHECK(cnrtFree(mlu_input_ptr));
    }
  }
  void setInputDataAndDimensions(void *cpu_input_ptr,
                                 const std::vector<magicmind::IRTTensor *> &inputs,
                                 const magicmind::Dims &dim,
                                 const int index);

 private:
  void *mlu_input_ptr = nullptr;
  int byte_size;
};

void inputManager::setInputDataAndDimensions(void *cpu_input_ptr,
                                             const std::vector<magicmind::IRTTensor *> &inputs,
                                             const magicmind::Dims &dim,
                                             const int index) {
  CNRT_CHECK(cnrtMemcpy(this->mlu_input_ptr, cpu_input_ptr, this->byte_size, cnrtMemcpyHostToDev));
  inputs[index]->SetDimensions(dim);
  inputs[index]->SetData(this->mlu_input_ptr);
}
#endif  // SAMPLES_CC_BASIC_SAMPLES_SAMPLE_PLUGINOP_COMMON_HELPER_FUNC_H_

