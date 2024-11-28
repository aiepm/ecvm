#include <ecvm/blocks/patch_embedding.hpp>
#include <torch/nn/options/conv.h>

PatchEmbeddingOptions::PatchEmbeddingOptions() = default;

auto PatchEmbeddingOptions::ImgSize(int x) -> PatchEmbeddingOptions& {
  this->img_size = x;
  return *this;
}

auto PatchEmbeddingOptions::EmbedDim(int x) -> PatchEmbeddingOptions& {
  this->embed_dim = x;
  return *this;
}

auto PatchEmbeddingOptions::PatchSize(int x) -> PatchEmbeddingOptions& {
  this->patch_size = x;
  return *this;
}

auto PatchEmbeddingOptions::InChannels(int x) -> PatchEmbeddingOptions& {
  this->in_channels = x;
  return *this;
}

PatchEmbedding::PatchEmbedding(const PatchEmbeddingOptions &ops) {
  int side_size = ops.img_size / ops.patch_size;
  this->n_patches = side_size * side_size;

  this->projection = torch::nn::Conv2d(torch::nn::Conv2dOptions(ops.in_channels, ops.embed_dim, ops.patch_size).stride(ops.patch_size));
}

PatchEmbedding::PatchEmbedding() = default;

auto PatchEmbedding::forward(torch::Tensor x) -> torch::Tensor {
  x = this->projection(x);
  x = x.flatten();
  x = x.transpose(1, 2);
  return x;
}
