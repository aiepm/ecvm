#ifndef VIT_HPP
#define VIT_HPP

#include "ecvm/blocks/patch_embedding.hpp"
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>
#include <torch/torch.h>

struct ViTOptions {
  int img_size=32;
  int patch_size=4;
  int in_channels=3;
  int num_classes=10;
  int embed_dim=64;
  int num_heads=4;
  int mlp_dim=128;
  int num_layers=1;
  int mlp_head_dim=512;
  int mlp_head_layers=4;
  double dropout_rate=0.1;

  ViTOptions();

  auto ImgSize(int x) -> ViTOptions&;
  auto PatchSize(int x) -> ViTOptions&;
  auto InChannels(int x) -> ViTOptions&;
  auto NumClasses(int x) -> ViTOptions&;
  auto EmbedDim(int x) -> ViTOptions&;
  auto NumHeads(int x) -> ViTOptions&;
  auto MLPDim(int x) -> ViTOptions&;
  auto NumLayers(int x) -> ViTOptions&;
  auto MLPHeadDim(int x) -> ViTOptions&;
  auto MLPHeadLayers(int x) -> ViTOptions&;
  auto DropoutRate(double x) -> ViTOptions&;
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
