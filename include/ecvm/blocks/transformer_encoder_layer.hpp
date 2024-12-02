#ifndef TRANSFORMER_ENCODER_LAYER_HPP
#define TRANSFORMER_ENCODER_LAYER_HPP

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

struct TransformerEncoderLayerOptions {
  int embed_dim = 64;
  int num_heads = 4;
  int mlp_dim = 128;
  double dropout_rate = 0.1;

  TransformerEncoderLayerOptions();

  auto EmbedDim(int x) -> TransformerEncoderLayerOptions&;
  auto NumHeads(int x) -> TransformerEncoderLayerOptions&;
  auto MLPDim(int x) -> TransformerEncoderLayerOptions&;
  auto DropoutRate(double x) -> TransformerEncoderLayerOptions&;
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
