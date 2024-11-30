#ifndef PATCH_EMBEDDING_HPP
#define PATCH_EMBEDDING_HPP

#include <torch/torch.h>

struct PatchEmbeddingOptions {
  int img_size = 32;
  int patch_size = 4;
  int in_channels = 3;
  int embed_dim = 64;

  PatchEmbeddingOptions();
  auto ImgSize(int x) -> PatchEmbeddingOptions&;
  auto PatchSize(int x) -> PatchEmbeddingOptions&;
  auto InChannels(int x) -> PatchEmbeddingOptions&;
  auto EmbedDim(int x) -> PatchEmbeddingOptions&;
};

struct PatchEmbedding : torch::nn::Module {
  int n_patches;
  torch::nn::Conv2d projection = nullptr;

  PatchEmbedding();
  PatchEmbedding(const PatchEmbeddingOptions&);
  auto forward(torch::Tensor x) -> torch::Tensor;
};

#endif // !PATCH_EMBEDDING_HPP
