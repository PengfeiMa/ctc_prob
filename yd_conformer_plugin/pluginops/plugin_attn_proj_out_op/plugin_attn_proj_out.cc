/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "plugin_attn_proj_out.h"
#include <cnnl_extra.h>
#include <cnrt.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>

PluginTransformerAttnProjOutKernel::PluginTransformerAttnProjOutKernel() {
  cnnlCreate(&handle_);
  cnnlCreateTensorDescriptor(&input_desc_);
  cnnlCreateTensorDescriptor(&residual_desc_);
  cnnlCreateTensorDescriptor(&q_filter_desc_);
  cnnlCreateTensorDescriptor(&q_bias_desc_);
  cnnlCreateTensorDescriptor(&q_out_desc_);
  cnnlCreateTransformerAttnProjDescriptor(&op_desc_);
};

PluginTransformerAttnProjOutKernel::~PluginTransformerAttnProjOutKernel() {
  cnnlDestroyTensorDescriptor(input_desc_);
  cnnlDestroyTensorDescriptor(residual_desc_);
  cnnlDestroyTensorDescriptor(q_filter_desc_);
  cnnlDestroyTensorDescriptor(q_bias_desc_);
  cnnlDestroyTensorDescriptor(q_out_desc_);
  cnnlDestroyTransformerAttnProjDescriptor(op_desc_);
  cnnlDestroy(handle_);
};

magicmind::Status PluginTransformerAttnProjOutKernel::SetLocalVar(
    magicmind::INodeResource *context) {
  // get input/mask/output dtype and check
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("input", &input_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("residual", &residual_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("q_filter", &q_filter_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("q_bias", &q_bias_dtype_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataType("q_out", &q_out_dtype_));

  // input
  if (input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "input data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(input_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // residual
  if (residual_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "input data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(residual_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // q_filter
  if (q_filter_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "k_filter data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(q_filter_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // q_bias
  if (q_bias_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "k_bias data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(q_bias_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // q_out
  if (q_out_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp = "k_out data type is invalid，should be fp16，but " +
                       magicmind::TypeEnumToString(q_out_dtype_) +
                       "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  // get input/output shape and check
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("input", &input_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("residual", &residual_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("q_filter", &q_filter_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("q_bias", &q_bias_shape_));
  ATTN_PROJ_MM_CHECK(context->GetTensorShape("q_out", &q_out_shape_));
  ATTN_PROJ_MM_CHECK(context->GetAttr("head_num", &head_num_));
  ATTN_PROJ_MM_CHECK(context->GetAttr("trans_out", &trans_out_));

  bool has_bias = true;
  bool is_pack_mode = false;
  int packed_max_seq_len = 0;
  bool store_layernorm_result = false;
  float alpha = 1.0;
  float beta = 1.0;
  float layernorm_eps = 0;

  batch_ = input_shape_[0];
  seq_len_ = input_shape_[1];
  input_size_ = input_shape_[2];
  hidden_size_ = q_filter_shape_[0];
  head_size_ = hidden_size_ / head_num_;
  int input_dim[3] = {batch_, seq_len_, input_size_};
  int residual_dim[3] = {(int)(residual_shape_[0]), (int)(residual_shape_[1]),
                         (int)(residual_shape_[2])};
  int q_filter_dim[2] = {hidden_size_, input_size_};
  int q_bias_dim[1] = {hidden_size_};
  int q_out_dim[3] = {batch_, seq_len_, hidden_size_};

  if (input_dtype_ == magicmind::DataType::FLOAT16) {
    dtype_ = CNNL_DTYPE_HALF;
  }
  if (input_dtype_ == magicmind::DataType::FLOAT32) {
    dtype_ = CNNL_DTYPE_FLOAT;
  }

  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      input_desc_, CNNL_LAYOUT_ARRAY, dtype_, input_shape_.size(), input_dim));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(residual_desc_, CNNL_LAYOUT_ARRAY,
                                          dtype_, residual_shape_.size(),
                                          residual_dim));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(q_filter_desc_, CNNL_LAYOUT_ARRAY,
                                          dtype_, q_filter_shape_.size(),
                                          q_filter_dim));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(q_bias_desc_, CNNL_LAYOUT_ARRAY,
                                          dtype_, q_bias_shape_.size(),
                                          q_bias_dim));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      q_out_desc_, CNNL_LAYOUT_ARRAY, dtype_, q_out_shape_.size(), q_out_dim));
  ATTN_CNNL_CHECK(cnnlSetTransformerAttnProjDescriptor(
      op_desc_, CNNL_TRANSFORMER_NO_LAYERNORM_WITH_RESIDUAL, nullptr, dtype_,
      q_has_value_, k_has_value_, v_has_value_, has_bias, is_pack_mode,
      packed_max_seq_len, trans_out_, store_layernorm_result, alpha, beta,
      layernorm_eps));

  return magicmind::Status::OK();
}

size_t PluginTransformerAttnProjOutKernel::GetWorkspaceSize(
    magicmind::INodeResource *context) {
  cnnlGetTransformerAttnProjWorkspaceSize(handle_, op_desc_, nullptr,
                                          input_desc_, q_filter_desc_,
                                          q_out_desc_, &workspace_size_);
  return workspace_size_;
}

magicmind::Status PluginTransformerAttnProjOutKernel::Enqueue(
    magicmind::INodeResource *context) {
  ATTN_PROJ_MM_CHECK(context->GetQueue(&queue_));
  ATTN_CNNL_CHECK(cnnlSetQueue(handle_, queue_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("input", &input_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("residual", &residual_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("q_filter", &q_filter_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("q_bias", &q_bias_addr_));
  ATTN_PROJ_MM_CHECK(context->GetTensorDataPtr("q_out", &q_out_addr_));
  ATTN_PROJ_MM_CHECK(context->GetWorkspace(&workspace_in_mlu_));

  ATTN_CNNL_CHECK(cnnlTransformerAttnProj(
      handle_, op_desc_, nullptr, input_desc_, input_addr_, residual_desc_,
      residual_addr_, q_filter_desc_, q_filter_addr_, k_filter_desc_,
      k_filter_addr_, v_filter_desc_, v_filter_addr_, q_bias_desc_,
      q_bias_addr_, k_bias_desc_, k_bias_addr_, v_bias_desc_, v_bias_addr_,
      nullptr, nullptr, layernorm_scale_desc_, layernorm_scale_addr_,
      layernorm_bias_desc_, layernorm_bias_addr_, workspace_in_mlu_,
      workspace_size_, q_out_desc_, q_out_addr_, k_out_desc_, k_out_addr_,
      v_out_desc_, v_out_addr_, nullptr, nullptr));

  return magicmind::Status::OK();
}
