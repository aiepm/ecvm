#include <ecvm/blocks/transformer_encoder_layer.hpp>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/options/normalization.h>

TransformerEncoderLayerOptions::TransformerEncoderLayerOptions() = default;

TransformerEncoderLayerImpl::TransformerEncoderLayerImpl() = default;

TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(const TransformerEncoderLayerOptions &ops) {
  this->ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)ops.embed_dim()})));
  this->ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)ops.embed_dim()})));
  this->mhsa = register_module("mhsa", torch::nn::MultiheadAttention(
      torch::nn::MultiheadAttentionOptions(ops.embed_dim(), ops.num_heads()).dropout(ops.dropout_rate())
  ));
  this->mlp = register_module("mlp", torch::nn::Sequential(
      torch::nn::Linear(ops.embed_dim(), ops.mlp_dim()),
      torch::nn::GELU(),
      torch::nn::Linear(ops.mlp_dim(), ops.embed_dim()),
      torch::nn::Dropout(ops.dropout_rate())
  ));
}

auto TransformerEncoderLayerImpl::forward(torch::Tensor x) -> torch::Tensor {
  auto x_ln = this->ln1(x);
  x_ln = x_ln.transpose(0, 1);
  auto [mhsa_output, _] = this->mhsa(x_ln, x_ln, x_ln);
  x = x + mhsa_output.transpose(0, 1);
  x_ln = this->ln2(x);
  x = x + this->mlp->forward(x_ln);
  return x;
}
