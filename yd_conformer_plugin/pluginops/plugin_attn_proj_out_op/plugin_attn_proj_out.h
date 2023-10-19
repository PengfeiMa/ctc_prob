/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#ifndef PLUGIN_ATTN_PROJ_OUT_H_
#define PLUGIN_ATTN_PROJ_OUT_H_
#include <cnnl_extra.h>
#include <iostream>
#include "common/macros.h"
#include "mm_plugin.h"

// 1. register custom op
#define ATTN_CNNL_CHECK(cnnl_api_call) \
  PLUGIN_CNNL_CHECK("[PluginATTNProjOut Internel]", cnnl_api_call)

#define ATTN_PROJ_MM_CHECK(mm_api_call) \
  PLUGIN_MM_CHECK("[PluginATTNProjOut Internel]", mm_api_call)

namespace magicmind {

Status ATTNProjectOutDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  std::vector<int64_t> q_filter_shape;
  int64_t head_num;
  int64_t trans_out;
  ATTN_PROJ_MM_CHECK(context->GetShape("input", &input_shape));
  ATTN_PROJ_MM_CHECK(context->GetShape("q_filter", &q_filter_shape));
  ATTN_PROJ_MM_CHECK(context->GetAttr("head_num", &head_num));
  ATTN_PROJ_MM_CHECK(context->GetAttr("trans_out", &trans_out));
  int64_t batch = input_shape[0];
  int64_t seq = input_shape[1];
  int64_t hidden_size = input_shape[2];
  int64_t head_size = hidden_size / head_num;
  if (trans_out) {
    context->SetShape("q_out",
                      std::vector<int64_t>{batch, head_num, seq, head_size});
  } else {
    context->SetShape("q_out", std::vector<int64_t>{batch, seq, hidden_size});
  }

  return Status::OK();
}

PLUGIN_REGISTER_OP("TRANSFORMER_ATTN_PROJ_OUT")
    .Input("input")
    .TypeConstraint("T")
    .Input("q_filter")
    .TypeConstraint("T")
    .Input("q_bias")
    .TypeConstraint("T")
    .Input("residual")
    .TypeConstraint("T")
    .Output("q_out")
    .TypeConstraint("T")
    .Param("T")
    .Type("type")
    .Allowed({magicmind::DataType::FLOAT16, magicmind::DataType::FLOAT32})
    .Param("head_num")
    .Type("int")
    .Param("trans_out")
    .Type("int")
    .ShapeFn(ATTNProjectOutDoShapeInfer);
}  // namespace magicmind

// 2.create plugin kernel
class PluginTransformerAttnProjOutKernel : public magicmind::IPluginKernel {
 public:
  // check kernel param
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  // set plugin workspace
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  PluginTransformerAttnProjOutKernel();
  ~PluginTransformerAttnProjOutKernel();

 private:
  void *input_addr_ = nullptr;
  void *residual_addr_ = nullptr;
  void *layernorm_scale_addr_ = nullptr;
  void *layernorm_bias_addr_ = nullptr;
  void *q_filter_addr_ = nullptr;
  void *q_bias_addr_ = nullptr;
  void *k_filter_addr_ = nullptr;
  void *k_bias_addr_ = nullptr;
  void *v_filter_addr_ = nullptr;
  void *v_bias_addr_ = nullptr;
  void *k_out_addr_ = nullptr;
  void *v_out_addr_ = nullptr;
  void *q_out_addr_ = nullptr;

  int batch_;
  int seq_len_;
  int input_size_;
  int hidden_size_;
  int64_t head_num_;
  int head_size_;
  int64_t trans_out_;
  int64_t q_has_value_ = 1;
  int64_t k_has_value_ = 0;
  int64_t v_has_value_ = 0;

  size_t workspace_size_;

  magicmind::DataType input_dtype_;
  magicmind::DataType residual_dtype_;
  magicmind::DataType layernorm_scale_dtype_;
  magicmind::DataType layernorm_bias_dtype_;
  magicmind::DataType q_filter_dtype_;
  magicmind::DataType q_bias_dtype_;
  magicmind::DataType k_filter_dtype_;
  magicmind::DataType k_bias_dtype_;
  magicmind::DataType v_filter_dtype_;
  magicmind::DataType v_bias_dtype_;
  magicmind::DataType q_out_dtype_;
  magicmind::DataType k_out_dtype_;
  magicmind::DataType v_out_dtype_;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> residual_shape_;
  std::vector<int64_t> layernorm_scale_shape_;
  std::vector<int64_t> layernorm_bias_shape_;
  std::vector<int64_t> q_filter_shape_;
  std::vector<int64_t> q_bias_shape_;
  std::vector<int64_t> k_filter_shape_;
  std::vector<int64_t> k_bias_shape_;
  std::vector<int64_t> v_filter_shape_;
  std::vector<int64_t> v_bias_shape_;
  std::vector<int64_t> q_out_shape_;
  std::vector<int64_t> k_out_shape_;
  std::vector<int64_t> v_out_shape_;

  cnrtQueue_t queue_;
  cnnlHandle_t handle_;
  cnnlDataType_t dtype_;
  cnnlTensorDescriptor_t input_desc_ = nullptr;
  cnnlTensorDescriptor_t residual_desc_ = nullptr;
  cnnlTensorDescriptor_t layernorm_scale_desc_ = nullptr;
  cnnlTensorDescriptor_t layernorm_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t q_filter_desc_ = nullptr;
  cnnlTensorDescriptor_t q_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t k_filter_desc_ = nullptr;
  cnnlTensorDescriptor_t k_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t v_filter_desc_ = nullptr;
  cnnlTensorDescriptor_t v_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t q_out_desc_ = nullptr;
  cnnlTensorDescriptor_t k_out_desc_ = nullptr;
  cnnlTensorDescriptor_t v_out_desc_ = nullptr;
  cnnlTransformerAttnProjDescriptor_t op_desc_ = nullptr;
  void *workspace_in_mlu_ = nullptr;
};

// 3.register kernel
class PluginTransformerAttnProjOutKernelFactory
    : public magicmind::IPluginKernelFactory {
 public:
  // rewrite create
  magicmind::IPluginKernel *Create() override {
    return new PluginTransformerAttnProjOutKernel();
  }
  ~PluginTransformerAttnProjOutKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(
    CreatePluginKernelDefBuilder("TRANSFORMER_ATTN_PROJ_OUT").DeviceType("MLU"),
    PluginTransformerAttnProjOutKernelFactory);
}  // namespace magicmind
#endif  // PLUGIN_ATTN_PROJ_OUT_H_
