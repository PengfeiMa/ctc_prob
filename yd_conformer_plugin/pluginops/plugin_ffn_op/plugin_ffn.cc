#include "plugin_ffn.h"
#include <cnnl_extra.h>
#include <iostream>
#include "common/macros.h"

#define FFN_CNNL_CHECK(cnnl_api_call) \
  PLUGIN_CNNL_CHECK("[PluginFFN Internel]", cnnl_api_call)

#define FFN_MM_CHECK(mm_api_call) \
  PLUGIN_MM_CHECK("[PluginFFN Internel]", mm_api_call)

PluginFeedForwardKernel::PluginFeedForwardKernel() {
  cnnlCreate(&handle_);
  cnnlCreateTensorDescriptor(&input_desc_);
  cnnlCreateTensorDescriptor(&fc1_weight_desc_);
  cnnlCreateTensorDescriptor(&fc1_bias_desc_);
  cnnlCreateTensorDescriptor(&fc2_weight_desc_);
  cnnlCreateTensorDescriptor(&fc2_bias_desc_);
  cnnlCreateTensorDescriptor(&norm_weight_desc_);
  cnnlCreateTensorDescriptor(&norm_bias_desc_);
  cnnlCreateTensorDescriptor(&output_desc_);

  cnnlCreateActivationDescriptor(&act_desc_);
  cnnlCreateTransformerFeedForwardDescriptor(&ffn_desc_);
}

PluginFeedForwardKernel::~PluginFeedForwardKernel() {
  cnnlDestroyTensorDescriptor(input_desc_);
  cnnlDestroyTensorDescriptor(fc1_weight_desc_);
  cnnlDestroyTensorDescriptor(fc1_bias_desc_);
  cnnlDestroyTensorDescriptor(fc2_weight_desc_);
  cnnlDestroyTensorDescriptor(fc2_bias_desc_);
  cnnlDestroyTensorDescriptor(norm_weight_desc_);
  cnnlDestroyTensorDescriptor(norm_bias_desc_);
  cnnlDestroyTensorDescriptor(output_desc_);

  cnnlDestroyActivationDescriptor(act_desc_);
  cnnlDestroyTransformerFeedForwardDescriptor(ffn_desc_);
  cnnlDestroy(handle_);
}

static std::vector<int> convertI64ToI32Vec(std::vector<int64_t> vec) {
  std::vector<int> res(vec.begin(), vec.end());
  return res;
}

magicmind::Status PluginFeedForwardKernel::SetLocalVar(
    magicmind::INodeResource *context) {
  // get input/output dtype and check
  FFN_MM_CHECK(context->GetTensorDataType("input", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("fc1_weight", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("fc1_bias", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("fc2_weight", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("fc2_bias", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("norm_weight", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("norm_bias", &input_dtype_));
  FFN_MM_CHECK(context->GetTensorDataType("output", &output_dtype_));

  if (input_dtype_ != magicmind::DataType::FLOAT32 &&
      input_dtype_ != magicmind::DataType::FLOAT16) {
    std::string temp =
        "Input data type is invalid, should be fp32 or fp16, but " +
        magicmind::TypeEnumToString(input_dtype_) + "is received.";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }
  if (input_dtype_ != output_dtype_) {
    std::string temp = "Input data type is " +
                       magicmind::TypeEnumToString(input_dtype_) +
                       "but output data type is " +
                       magicmind::TypeEnumToString(output_dtype_) + ".";
    magicmind::Status status(magicmind::error::Code::UNAVAILABLE, temp);
    return status;
  }

  FFN_MM_CHECK(context->GetTensorShape("input", &input_shape_));
  FFN_MM_CHECK(context->GetTensorShape("fc1_weight", &fc1_weight_shape_));
  FFN_MM_CHECK(context->GetTensorShape("fc1_bias", &fc1_bias_shape_));
  FFN_MM_CHECK(context->GetTensorShape("fc2_weight", &fc2_weight_shape_));
  FFN_MM_CHECK(context->GetTensorShape("fc2_bias", &fc2_bias_shape_));
  FFN_MM_CHECK(context->GetTensorShape("norm_weight", &norm_weight_shape_));
  FFN_MM_CHECK(context->GetTensorShape("norm_bias", &norm_bias_shape_));
  FFN_MM_CHECK(context->GetTensorShape("output", &output_shape_));
  FFN_MM_CHECK(context->GetAttr("inside_residual", &inside_residual_));
  FFN_MM_CHECK(context->GetAttr("compute_dtype", &compute_dtype_));
  FFN_MM_CHECK(context->GetAttr("use_hp_active", &use_hp_active_));

  if (input_dtype_ == magicmind::DataType::FLOAT16) {
    dtype = CNNL_DTYPE_HALF;
  }
  if (input_dtype_ == magicmind::DataType::FLOAT32) {
    dtype = CNNL_DTYPE_FLOAT;
  }

  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      input_desc_, CNNL_LAYOUT_ARRAY, dtype, input_shape_.size(),
      convertI64ToI32Vec(input_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      fc1_weight_desc_, CNNL_LAYOUT_ARRAY, dtype, fc1_weight_shape_.size(),
      convertI64ToI32Vec(fc1_weight_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      fc1_bias_desc_, CNNL_LAYOUT_ARRAY, dtype, fc1_bias_shape_.size(),
      convertI64ToI32Vec(fc1_bias_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      fc2_weight_desc_, CNNL_LAYOUT_ARRAY, dtype, fc2_weight_shape_.size(),
      convertI64ToI32Vec(fc2_weight_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      fc2_bias_desc_, CNNL_LAYOUT_ARRAY, dtype, fc2_bias_shape_.size(),
      convertI64ToI32Vec(fc2_bias_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      norm_weight_desc_, CNNL_LAYOUT_ARRAY, dtype, norm_weight_shape_.size(),
      convertI64ToI32Vec(norm_weight_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      norm_bias_desc_, CNNL_LAYOUT_ARRAY, dtype, norm_bias_shape_.size(),
      convertI64ToI32Vec(norm_bias_shape_).data()));
  FFN_CNNL_CHECK(cnnlSetTensorDescriptor(
      output_desc_, CNNL_LAYOUT_ARRAY, dtype, output_shape_.size(),
      convertI64ToI32Vec(output_shape_).data()));

  if (inside_residual_) {
    layernorm_structure_ = CNNL_TRANSFORMER_PRE_LAYERNORM_INSIDE_RESIDUAL;
  }

  FFN_CNNL_CHECK(cnnlSetTransformerFeedForwardDescriptor_v2(
      ffn_desc_, layernorm_eps_, alpha_, beta_, (cnnlDataType_t)compute_dtype_,
      layernorm_structure_));

  if (use_hp_active_) {
    act_pref_ = CNNL_ACTIVATION_HIGH_PRECISION;
  }

  FFN_CNNL_CHECK(cnnlSetActivationDescriptor_v5(act_desc_, act_mode_, act_pref_,
                                                (cnnlNanPropagation_t)0,
                                                act_coef_, 0, 0, 0, false));

  return magicmind::Status::OK();
}

size_t PluginFeedForwardKernel::GetWorkspaceSize(
    magicmind::INodeResource *context) {
  cnnlGetTransformerFeedForwardWorkspaceSize(
      handle_, ffn_desc_, input_desc_, fc1_weight_desc_, &ffn_workspace_size_);

  return ffn_workspace_size_;
}

magicmind::Status PluginFeedForwardKernel::Enqueue(
    magicmind::INodeResource *context) {
  FFN_MM_CHECK(context->GetQueue(&queue_));
  FFN_CNNL_CHECK(cnnlSetQueue(handle_, queue_));
  FFN_MM_CHECK(context->GetTensorDataPtr("input", &input_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("fc1_weight", &fc1_weight_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("fc1_bias", &fc1_bias_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("fc2_weight", &fc2_weight_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("fc2_bias", &fc2_bias_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("norm_weight", &norm_weight_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("norm_bias", &norm_bias_addr_));
  FFN_MM_CHECK(context->GetTensorDataPtr("output", &output_addr_));
  FFN_MM_CHECK(context->GetWorkspace(&ffn_workspace_));

  FFN_CNNL_CHECK(cnnlTransformerFeedForward(
      handle_, ffn_desc_, act_desc_, nullptr, input_desc_, input_addr_,
      fc1_weight_desc_, fc1_weight_addr_, fc1_bias_desc_, fc1_bias_addr_,
      fc2_weight_desc_, fc2_weight_addr_, fc2_bias_desc_, fc2_bias_addr_,
      norm_weight_desc_, norm_weight_addr_, norm_bias_desc_, norm_bias_addr_,
      ffn_workspace_, ffn_workspace_size_, output_desc_, output_addr_));

  return magicmind::Status::OK();
}
