#ifndef TRANSFORMER_ENCODER_LAYER_HPP
#define TRANSFORMER_ENCODER_LAYER_HPP

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

struct TransformerEncoderLayerOptions {
  TransformerEncoderLayerOptions();

  TORCH_ARG(size_t, embed_dim) = 64;
  TORCH_ARG(size_t, num_heads) = 4;
  TORCH_ARG(size_t, mlp_dim) = 128;
  TORCH_ARG(double, dropout_rate) = 0.1;
};

struct TransformerEncoderLayerImpl : torch::nn::Module {
  torch::nn::LayerNorm ln1 = nullptr, ln2 = nullptr;
  torch::nn::MultiheadAttention mhsa = nullptr;
  torch::nn::Sequential mlp;

  TransformerEncoderLayerImpl();
  TransformerEncoderLayerImpl(const TransformerEncoderLayerOptions&);

  auto forward(torch::Tensor x) -> torch::Tensor;
};

TORCH_MODULE(TransformerEncoderLayer);

#endif // !TRANSFORMER_ENCODER_LAYER_HPP
