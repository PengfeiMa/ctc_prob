/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_ATTN_H_
#define PLUGIN_ATTN_H_
#include <cnnl_extra.h>
#include <iostream>
#include "mm_plugin.h"
// 1. register custom op
namespace magicmind {

Status AttentionDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> query_shape;
  std::vector<int64_t> output_shape;
  context->GetShape("query",
                    &query_shape);  // [batch, head_num, seq_q, head_size]
  if (query_shape.size() != 4) {
    return magicmind::Status(
        magicmind::error::Code::INTERNAL,
        "The size of query shape in TRANSFORMER_ATTENTION is not equal to 4");
  }
  output_shape.push_back(query_shape[0]);
  output_shape.push_back(query_shape[2]);
  output_shape.push_back(query_shape[1]);
  output_shape.push_back(query_shape[3]);
  context->SetShape(
      "output", output_shape);  // [batch * beam, seq_q, head_num, head_size]
  return Status::OK();
}

PLUGIN_REGISTER_OP("TRANSFORMER_ATTENTION")
    .Input("query")
    .TypeConstraint("T")
    .Input("key")
    .TypeConstraint("T")
    .Input("value")
    .TypeConstraint("T")
    // .Input("mask").TypeConstraint("T")
    .Output("output")
    .TypeConstraint("T")
    .Param("T")
    .Type("type")
    .Allowed({magicmind::DataType::FLOAT32, magicmind::DataType::FLOAT16})
    .Param("has_mask")
    .Type("int")
    .Param("is_mul_factor_after_qk")
    .Type("int")
    .Param("query_factor")
    .Type("float")
    .Param("compute_dtype")
    .Type("int")
    .Param("use_hp_active")
    .Type("int")
    .ShapeFn(AttentionDoShapeInfer);
}  // namespace magicmind

// 2.create plugin kernel
class PluginTransformerAttentionKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  PluginTransformerAttentionKernel();
  ~PluginTransformerAttentionKernel();

 private:
  size_t attn_workspace_size_ = 0;
  void *workspace_ = nullptr;
  void *query_addr_ = nullptr;
  void *key_addr_ = nullptr;
  void *value_addr_ = nullptr;
  void *mask_addr_ = nullptr;
  void *output_addr_ = nullptr;
  magicmind::DataType input_dtype_;
  magicmind::DataType output_dtype_;
  std::vector<int64_t> query_shape_;
  std::vector<int64_t> key_shape_;
  std::vector<int64_t> value_shape_;
  std::vector<int64_t> mask_shape_;
  std::vector<int64_t> output_shape_;
  cnrtQueue_t queue_;
  cnnlHandle_t handle_;
  cnnlDataType_t dtype;
  int64_t has_mask_;
  int64_t is_mul_factor_after_qk_;
  float query_factor_;
  int64_t compute_dtype_;
  int64_t use_hp_active_;
  cnnlTensorDescriptor_t query_desc_ = nullptr;
  cnnlTensorDescriptor_t key_desc_ = nullptr;
  cnnlTensorDescriptor_t value_desc_ = nullptr;
  cnnlTensorDescriptor_t mask_desc_ = nullptr;
  cnnlTensorDescriptor_t output_desc_ = nullptr;
  cnnlTransformerAttentionDescriptor_t op_desc_ = nullptr;
};

// 3.register kernel
class PluginTransformerAttentionKernelFactory
    : public magicmind::IPluginKernelFactory {
 public:
  // rewrite create
  magicmind::IPluginKernel *Create() override {
    return new PluginTransformerAttentionKernel();
  }
  ~PluginTransformerAttentionKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(
    CreatePluginKernelDefBuilder("TRANSFORMER_ATTENTION").DeviceType("MLU"),
    PluginTransformerAttentionKernelFactory);
}  // namespace magicmind

#endif  // PLUGIN_ATTN_H_
