/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/
#include "plugin_attn.h"
#include <cnnl_extra.h>
#include <cnrt.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include "common/macros.h"

#define ATTN_CNNL_CHECK(cnnl_api_call) \
  PLUGIN_CNNL_CHECK("[PluginATTN Internel]", cnnl_api_call)

#define ATTN_MM_CHECK(mm_api_call) \
  PLUGIN_MM_CHECK("[PluginATTN Internel]", mm_api_call)

PluginTransformerAttentionKernel::PluginTransformerAttentionKernel() {
  cnnlCreate(&handle_);
  cnnlCreateTensorDescriptor(&query_desc_);
  cnnlCreateTensorDescriptor(&key_desc_);
  cnnlCreateTensorDescriptor(&value_desc_);
  // cnnlCreateTensorDescriptor(&mask_desc_);
  cnnlCreateTensorDescriptor(&output_desc_);
  cnnlCreateTransformerAttentionDescriptor(&op_desc_);
};

PluginTransformerAttentionKernel::~PluginTransformerAttentionKernel() {
  cnnlDestroyTensorDescriptor(query_desc_);
  cnnlDestroyTensorDescriptor(key_desc_);
  cnnlDestroyTensorDescriptor(value_desc_);
  // cnnlDestroyTensorDescriptor(mask_desc_);
  cnnlDestroyTensorDescriptor(output_desc_);
  cnnlDestroyTransformerAttentionDescriptor(op_desc_);
  cnnlDestroy(handle_);
};

static std::vector<int> convertI64ToI32Vec(std::vector<int64_t> vec) {
  std::vector<int> res(vec.begin(), vec.end());
  return res;
}

magicmind::Status PluginTransformerAttentionKernel::SetLocalVar(
    magicmind::INodeResource *context) {
  // get input/mask/output dtype and check
  ATTN_MM_CHECK(context->GetTensorDataType("query", &input_dtype_));
  ATTN_MM_CHECK(context->GetTensorDataType("output", &output_dtype_));

  // input
  if (input_dtype_ != magicmind::DataType::FLOAT32 &&
      input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp =
        "input data type is invalid, should be fp32 or fp16, but " +
        magicmind::TypeEnumToString(input_dtype_) + "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  // output
  if (input_dtype_ != output_dtype_) {
    std::string temp = "Input data type is " +
                       magicmind::TypeEnumToString(input_dtype_) +
                       "but output data type is " +
                       magicmind::TypeEnumToString(output_dtype_) + ".";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  if (input_dtype_ == magicmind::DataType::FLOAT16) {
    dtype = CNNL_DTYPE_HALF;
  }
  if (input_dtype_ == magicmind::DataType::FLOAT32) {
    dtype = CNNL_DTYPE_FLOAT;
  }

  ATTN_MM_CHECK(context->GetTensorShape("query", &query_shape_));
  ATTN_MM_CHECK(context->GetTensorShape("key", &key_shape_));
  ATTN_MM_CHECK(context->GetTensorShape("value", &value_shape_));
  // ATTN_MM_CHECK(context->GetTensorShape("mask", &mask_shape_));
  ATTN_MM_CHECK(context->GetTensorShape("output", &output_shape_));
  ATTN_MM_CHECK(context->GetAttr("has_mask", &has_mask_));
  ATTN_MM_CHECK(
      context->GetAttr("is_mul_factor_after_qk", &is_mul_factor_after_qk_));
  ATTN_MM_CHECK(context->GetAttr("query_factor", &query_factor_));
  ATTN_MM_CHECK(context->GetAttr("compute_dtype", &compute_dtype_));
  ATTN_MM_CHECK(context->GetAttr("use_hp_active", &use_hp_active_));

  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      query_desc_, CNNL_LAYOUT_ARRAY, dtype, query_shape_.size(),
      convertI64ToI32Vec(query_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      key_desc_, CNNL_LAYOUT_ARRAY, dtype, key_shape_.size(),
      convertI64ToI32Vec(key_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      value_desc_, CNNL_LAYOUT_ARRAY, dtype, value_shape_.size(),
      convertI64ToI32Vec(value_shape_).data()));
  // ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
  //     mask_desc_, CNNL_LAYOUT_ARRAY, dtype, mask_shape_.size(),
  //     convertI64ToI32Vec(mask_shape_).data()));
  ATTN_CNNL_CHECK(cnnlSetTensorDescriptor(
      output_desc_, CNNL_LAYOUT_ARRAY, dtype, output_shape_.size(),
      convertI64ToI32Vec(output_shape_).data()));

  cnnlAttentionMaskMode_t mask_mode;
  if (has_mask_) {
    mask_mode = CNNL_ATTN_MASK_NHTT;
  } else {
    mask_mode = CNNL_ATTN_MASK_NONE;
  }
  cnnlActivationPreference_t act_pref;
  if (use_hp_active_) {
    act_pref = CNNL_ACTIVATION_HIGH_PRECISION;
  } else {
    act_pref = CNNL_ACTIVATION_FAST;
  }

  ATTN_CNNL_CHECK(cnnlSetTransformerAttentionDescriptor(
      op_desc_, (cnnlDataType_t)compute_dtype_, act_pref, mask_mode, false, 0,
      query_factor_, is_mul_factor_after_qk_, false));

  return magicmind::Status::OK();
}

size_t PluginTransformerAttentionKernel::GetWorkspaceSize(
    magicmind::INodeResource *context) {
  cnnlGetTransformerAttentionWorkspaceSize(handle_, op_desc_, nullptr,
                                           query_desc_, key_desc_, value_desc_,
                                           &attn_workspace_size_);
  return attn_workspace_size_;
}

magicmind::Status PluginTransformerAttentionKernel::Enqueue(
    magicmind::INodeResource *context) {
  ATTN_MM_CHECK(context->GetQueue(&queue_));
  ATTN_CNNL_CHECK(cnnlSetQueue(handle_, queue_));
  ATTN_MM_CHECK(context->GetTensorDataPtr("query", &query_addr_));
  ATTN_MM_CHECK(context->GetTensorDataPtr("key", &key_addr_));
  ATTN_MM_CHECK(context->GetTensorDataPtr("value", &value_addr_));
  // ATTN_MM_CHECK(context->GetTensorDataPtr("mask", &mask_addr_));
  ATTN_MM_CHECK(context->GetTensorDataPtr("output", &output_addr_));
  ATTN_MM_CHECK(context->GetWorkspace(&workspace_));

  ATTN_CNNL_CHECK(cnnlTransformerAttention(
      handle_, op_desc_, nullptr, query_desc_, query_addr_, key_desc_,
      key_addr_, value_desc_, value_addr_,
      nullptr,  // mask_desc_
      nullptr,  // mask_addr_
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      workspace_, attn_workspace_size_, nullptr, nullptr, output_desc_,
      output_addr_));

  return magicmind::Status::OK();
}
