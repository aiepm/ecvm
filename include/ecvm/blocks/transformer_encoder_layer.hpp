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

struct TransformerEncoderLayer : torch::nn::Module {
  torch::nn::LayerNorm ln1, ln2;
  torch::nn::MultiheadAttention mhsa;
  torch::nn::Sequential mlp;

  TransformerEncoderLayer();
  TransformerEncoderLayer(const TransformerEncoderLayerOptions&);

  auto forward(torch::Tensor x) -> torch::Tensor;
};

#endif // !TRANSFORMER_ENCODER_LAYER_HPP
