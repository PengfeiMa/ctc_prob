#ifndef PLUGIN_FFN_H_
#define PLUGIN_FFN_H_
#include <cnnl_extra.h>
#include "cnrt.h"
#include "mm_plugin.h"
// 1.register custom op
namespace magicmind {

Status FeedForwardDoShapeInfer(IShapeInferResource *context) {
  std::vector<int64_t> input_shape;
  context->GetShape("input", &input_shape);
  context->SetShape("output", input_shape);
  return Status::OK();
}

PLUGIN_REGISTER_OP("TRANSFORMER_FEED_FORWARD")
    .Input("input")
    .TypeConstraint("T")
    .Input("fc1_weight")
    .TypeConstraint("T")
    .Input("fc1_bias")
    .TypeConstraint("T")
    .Input("fc2_weight")
    .TypeConstraint("T")
    .Input("fc2_bias")
    .TypeConstraint("T")
    .Input("norm_weight")
    .TypeConstraint("T")
    .Input("norm_bias")
    .TypeConstraint("T")
    .Output("output")
    .TypeConstraint("T")
    .Param("T")
    .Type("type")
    .Allowed({DataType::FLOAT16, DataType::FLOAT32})
    .Param("pre_layernorm")
    .Type("int")
    .Param("post_layernorm")
    .Type("int")
    .Param("inside_residual")
    .Type("int")
    .Param("compute_dtype")
    .Type("int")
    .Param("use_hp_active")
    .Type("int")
    .ShapeFn(FeedForwardDoShapeInfer);
}  // namespace magicmind

// 2.create plugin kernel
class PluginFeedForwardKernel : public magicmind::IPluginKernel {
 public:
  //完成参数检查和用户自定义成员变量的初始化
  magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
  //获取ffn 操作运行时所需的额外内存（如果有的话）
  size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
  //执行运算
  magicmind::Status Enqueue(magicmind::INodeResource *context) override;
  PluginFeedForwardKernel();
  ~PluginFeedForwardKernel();

 private:
  int64_t inside_residual_;
  int64_t compute_dtype_;
  int64_t use_hp_active_;
  float layernorm_eps_ = 1e-12;
  float alpha_ = 1.0;
  float beta_ = 1.0;
  cnnlTransformerLayernormResidualStructure_t layernorm_structure_ =
      CNNL_TRANSFORMER_NO_LAYERNORM_NO_RESIDUAL;
  cnnlActivationMode_t act_mode_ = CNNL_ACTIVATION_RELU;
  cnnlActivationPreference_t act_pref_ = CNNL_ACTIVATION_FAST;
  float act_coef_ = 0.0;
  int64_t allow_tf32_ = 0;

  void *input_addr_ = nullptr;
  void *fc1_weight_addr_ = nullptr;
  void *fc1_bias_addr_ = nullptr;
  void *fc2_weight_addr_ = nullptr;
  void *fc2_bias_addr_ = nullptr;
  void *norm_weight_addr_ = nullptr;
  void *norm_bias_addr_ = nullptr;
  void *output_addr_ = nullptr;

  magicmind::DataType input_dtype_;
  magicmind::DataType output_dtype_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> fc1_weight_shape_;
  std::vector<int64_t> fc1_bias_shape_;
  std::vector<int64_t> fc2_weight_shape_;
  std::vector<int64_t> fc2_bias_shape_;
  std::vector<int64_t> norm_weight_shape_;
  std::vector<int64_t> norm_bias_shape_;

  std::vector<int64_t> output_shape_;
  cnrtQueue_t queue_;
  cnnlDataType_t dtype;

  cnnlTensorDescriptor_t input_desc_ = nullptr;
  cnnlTensorDescriptor_t fc1_weight_desc_ = nullptr;
  cnnlTensorDescriptor_t fc1_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t fc2_weight_desc_ = nullptr;
  cnnlTensorDescriptor_t fc2_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t norm_weight_desc_ = nullptr;
  cnnlTensorDescriptor_t norm_bias_desc_ = nullptr;
  cnnlTensorDescriptor_t output_desc_ = nullptr;
  cnnlActivationDescriptor_t act_desc_;
  cnnlTransformerFeedForwardDescriptor_t ffn_desc_;

  cnnlDeviceType_t device = CNNL_MLU_370_X;

  void *ffn_workspace_ = nullptr;
  size_t ffn_workspace_size_ = 0;
  cnnlHandle_t handle_;
};

// 3.register kernel
class PluginFeedForwardKernelFactory : public magicmind::IPluginKernelFactory {
 public:
  magicmind::IPluginKernel *Create() override {
    return new PluginFeedForwardKernel();
  }
  ~PluginFeedForwardKernelFactory() {}
};

namespace magicmind {
PLUGIN_REGISTER_KERNEL(
    CreatePluginKernelDefBuilder("TRANSFORMER_FEED_FORWARD").DeviceType("MLU"),
    PluginFeedForwardKernelFactory);
}  // namespace magicmind
#endif  // PLUGIN_FFN_H_
