#ifndef PATCH_EMBEDDING_HPP
#define PATCH_EMBEDDING_HPP

#include <torch/torch.h>

struct PatchEmbeddingOptions {
  PatchEmbeddingOptions();

  TORCH_ARG(size_t, img_size) = 32;
  TORCH_ARG(size_t, patch_size) = 4;
  TORCH_ARG(size_t, in_channels) = 3;
  TORCH_ARG(size_t, embed_dim) = 64;
};

struct PatchEmbeddingImpl : torch::nn::Module {
  int n_patches;
  torch::nn::Conv2d projection = nullptr;

  PatchEmbeddingImpl();
  PatchEmbeddingImpl(const PatchEmbeddingOptions&);
  auto forward(torch::Tensor x) -> torch::Tensor;
};

TORCH_MODULE(PatchEmbedding);

#endif // !PATCH_EMBEDDING_HPP
