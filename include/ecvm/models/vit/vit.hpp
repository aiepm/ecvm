#ifndef VIT_HPP
#define VIT_HPP

#include <ecvm/blocks/patch_embedding.hpp>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>
#include <torch/torch.h>

struct ViTOptions {
  ViTOptions();
  TORCH_ARG(size_t, img_size) = 32;
  TORCH_ARG(size_t, patch_size) = 4;
  TORCH_ARG(size_t, in_channels) = 3;
  TORCH_ARG(size_t, num_classes) = 10;
  TORCH_ARG(size_t, embed_dim) = 64;
  TORCH_ARG(size_t, num_heads) = 4;
  TORCH_ARG(size_t, mlp_dim) = 256;
  TORCH_ARG(size_t, num_layers) = 3;
  TORCH_ARG(size_t, mlp_head_dim) = 256;
  TORCH_ARG(size_t, mlp_head_layers) = 4;
  TORCH_ARG(double, dropout_rate) = 0.1;
};

struct ViT : torch::nn::Module {
  PatchEmbedding patch_embedding;
  torch::Tensor position_embeddings;
  torch::nn::Sequential transformer_layers, mlp_head;
  torch::nn::Flatten flatten;

  ViT(const ViTOptions &ops);
  auto forward(torch::Tensor x) -> torch::Tensor;
};

#endif // !VIT_HPP
