/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_ATTN_PROJ_IN_H_
#define PLUGIN_ATTN_PROJ_IN_H_
#include <cnnl_extra.h>
#include <iostream>
#include "common/macros.h"
#include "mm_plugin.h"

// 1. register custom op
#define ATTN_CNNL_CHECK(cnnl_api_call) \
  PLUGIN_CNNL_CHECK("[PluginATTNProjIN Internel]", cnnl_api_call)

#define ATTN_PROJ_MM_CHECK(mm_api_call) \
  PLUGIN_MM_CHECK("[PluginATTNProjIN Internel]", mm_api_call)

namespace magicmind {

Status ATTNProjectINDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  int64_t head_num;
  ATTN_PROJ_MM_CHECK(context->GetShape("input", &input_shape));
  ATTN_PROJ_MM_CHECK(context->GetAttr("head_num", &head_num));
  int64_t batch = input_shape[0];
  int64_t seq = input_shape[1];
  int64_t hidden_size = input_shape[2];
  int64_t head_size = hidden_size / head_num;
  std::vector<int64_t> output_shape = {batch, head_num, seq, head_size};
  context->SetShape("output", output_shape);
  return Status::OK();
}

PLUGIN_REGISTER_OP("TRANSFORMER_ATTN_PROJ_IN")
    .Input("input")
    .TypeConstraint("T")
    .Input("weight")
    .TypeConstraint("T")
    .Input("bias")
    .TypeConstraint("T")
    .Output("output")
    .TypeConstraint("T")
    .Param("T")
    .Type("type")
    .Allowed({magicmind::DataType::FLOAT16, magicmind::DataType::FLOAT32})
    .Param("head_num")
    .Type("int")
    .ShapeFn(ATTNProjectINDoShapeInfer);
}  // namespace magicmind

// 2.create plugin kernel
class PluginTransformerAttnProjINKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  PluginTransformerAttnProjINKernel();
  ~PluginTransformerAttnProjINKernel();

 private:
  void *input_addr_ = nullptr;
  void *weight_addr_ = nullptr;
  void *bias_addr_ = nullptr;
  void *output_addr_ = nullptr;

  int64_t head_num_;

  magicmind::DataType input_dtype_;
  magicmind::DataType weight_dtype_;
  magicmind::DataType bias_dtype_;
  magicmind::DataType output_dtype_;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> weight_shape_;
  std::vector<int64_t> bias_shape_;
  std::vector<int64_t> output_shape_;

  cnrtQueue_t queue_;
  cnnlHandle_t handle_;
  cnnlDataType_t dtype_;
  cnnlTensorDescriptor_t input_desc_ = nullptr;
  cnnlTensorDescriptor_t weight_desc_ = nullptr;
  cnnlTensorDescriptor_t bias_desc_ = nullptr;
  cnnlTensorDescriptor_t output_desc_ = nullptr;
  cnnlTransformerAttnProjDescriptor_t op_desc_ = nullptr;

  size_t workspace_size_;
  void *workspace_ = nullptr;
};

// 3.register kernel
class PluginTransformerAttnProjINKernelFactory
    : public magicmind::IPluginKernelFactory {
 public:
  // rewrite create
  magicmind::IPluginKernel *Create() override {
    return new PluginTransformerAttnProjINKernel();
  }
  ~PluginTransformerAttnProjINKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(
    CreatePluginKernelDefBuilder("TRANSFORMER_ATTN_PROJ_IN").DeviceType("MLU"),
    PluginTransformerAttnProjINKernelFactory);
}  // namespace magicmind
#endif  // PLUGIN_ATTN_PROJ_KV_H_
