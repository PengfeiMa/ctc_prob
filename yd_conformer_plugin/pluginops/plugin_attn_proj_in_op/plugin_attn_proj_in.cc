/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "plugin_attn_proj_in.h"
#include <cnnl_extra.h>
#include <cnrt.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>

PluginTransformerAttnProjINKernel::PluginTransformerAttnProjINKernel() {
  cnnlCreate(&handle_);
  cnnlCreateTensorDescriptor(&input_desc_);
  cnnlCreateTensorDescriptor(&weight_desc_);
  cnnlCreateTensorDescriptor(&bias_desc_);
  cnnlCreateTensorDescriptor(&output_desc_);
  cnnlCreateTransformerAttnProjDescriptor(&op_desc_);
};

PluginTransformerAttnProjINKernel::~PluginTransformerAttnProjINKernel() {
  cnnlDestroyTensorDescriptor(input_desc_);
  cnnlDestroyTensorDescriptor(weight_desc_);
  cnnlDestroyTensorDescriptor(bias_desc_);
  cnnlDestroyTensorDescriptor(output_desc_);
  cnnlDestroyTransformerAttnProjDescriptor(op_desc_);
  cnnlDestroy(handle_);
};

static std::vector<int> convertI64ToI32Vec(std::vector<int64_t> vec) {
  std::vector<int> res(vec.begin(), vec.end());
  return res;
}

magicmind::Status PluginTransformerAttnProjINKernel::SetLocalVar(
    magicmind::INodeResource *context) {
  // get input/mask/output dtype and check
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("input", &input_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("weight", &weight_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("bias", &bias_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("output", &output_dtype_));

  // input
  if (input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "input data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(input_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // weight
  if (weight_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "input data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(weight_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // bias
  if (bias_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "input data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(bias_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // output
  if (output_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "output data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(output_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // get input/output shape and check
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("input", &input_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("weight", &weight_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("bias", &bias_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("output", &output_shape_));
  ATTN_PROJ_MM_CHECK(context->GetAttr("head_num", &head_num_));

  if (input_dtype_ == magicmind::DataType::FLOAT16) {
    dtype_ = CNNL_DTYPE_HALF;
  }
  if (input_dtype_ == magicmind::DataType::FLOAT32) {
    dtype_ = CNNL_DTYPE_FLOAT;
  }

  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      input_desc_, CNNL_LAYOUT_ARRAY, dtype_, input_shape_.size(),
      convertI64ToI32Vec(input_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      weight_desc_, CNNL_LAYOUT_ARRAY, dtype_, weight_shape_.size(),
      convertI64ToI32Vec(weight_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      bias_desc_, CNNL_LAYOUT_ARRAY, dtype_, bias_shape_.size(),
      convertI64ToI32Vec(bias_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      output_desc_, CNNL_LAYOUT_ARRAY, dtype_, output_shape_.size(),
      convertI64ToI32Vec(output_shape_).data()));

  bool q_has_value = true;
  bool k_has_value = false;
  bool v_has_value = false;
  bool has_bias = true;
  bool is_pack_mode = false;
  int packed_max_seq_len = 0;
  bool trans_out = true;
  bool store_layernorm_result = false;
  float alpha = 1.0;
  float beta = 1.0;
  float layernorm_eps = 1e-12;

  ATTN_CNNL_CHECK(cnnlSetTransformerAttnProjDescriptor(
      op_desc_, CNNL_TRANSFORMER_NO_LAYERNORM_NO_RESIDUAL, nullptr, dtype_,
      q_has_value, k_has_value, v_has_value, has_bias, is_pack_mode,
      packed_max_seq_len, trans_out, store_layernorm_result, alpha, beta,
      layernorm_eps));

  return magicmind::Status::OK();
}

size_t PluginTransformerAttnProjINKernel::GetWorkspaceSize(
    magicmind::INodeResource *context) {
  cnnlGetTransformerAttnProjWorkspaceSize(handle_, op_desc_, nullptr,
                                          input_desc_, weight_desc_,
                                          output_desc_, &workspace_size_);
  return workspace_size_;
}

magicmind::Status PluginTransformerAttnProjINKernel::Enqueue(
    magicmind::INodeResource *context) {
  ATTN_PROJ_MM_CHECK(context->GetQueue(&queue_));
  ATTN_CNNL_CHECK(cnnlSetQueue(handle_, queue_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("input", &input_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("weight", &weight_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("bias", &bias_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("output", &output_addr_));
  ATTN_PROJ_MM_CHECK(context->GetWorkspace(&workspace_));

  ATTN_CNNL_CHECK(cnnlTransformerAttnProj(
      handle_, op_desc_, nullptr, input_desc_, input_addr_, nullptr,
      nullptr,                     // residual
      weight_desc_, weight_addr_,  // q
      nullptr, nullptr,            // k
      nullptr, nullptr,            // v
      bias_desc_, bias_addr_,      // q bias
      nullptr, nullptr,            // k bias
      nullptr, nullptr,            // v bias
      nullptr, nullptr,            // valid_token
      nullptr, nullptr,            // layernorm scale
      nullptr, nullptr,            // layernorm bias
      workspace_, workspace_size_, output_desc_, output_addr_,  // q out
      nullptr, nullptr,                                         // k out
      nullptr, nullptr,                                         // v out
      nullptr, nullptr                                          // layernorm out
      ));

  return magicmind::Status::OK();
}
