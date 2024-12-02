#include <ecvm/blocks/patch_embedding.hpp>
#include <torch/nn/options/conv.h>

PatchEmbeddingOptions::PatchEmbeddingOptions() = default;

PatchEmbeddingImpl::PatchEmbeddingImpl(const PatchEmbeddingOptions &ops) {
  int side_size = ops.img_size() / ops.patch_size();
  this->n_patches = side_size * side_size;

  this->projection = register_module("projection", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(
          ops.in_channels(),
          ops.embed_dim(),
          ops.patch_size()
        ).stride(ops.patch_size())
  ));
}

PatchEmbeddingImpl::PatchEmbeddingImpl() = default;

auto PatchEmbeddingImpl::forward(torch::Tensor x) -> torch::Tensor {
  x = this->projection(x);
  x = x.flatten(2);
  x = x.transpose(1, 2);
  return x;
}
