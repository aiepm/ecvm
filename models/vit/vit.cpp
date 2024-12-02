#include <ecvm/blocks/patch_embedding.hpp>
#include <ecvm/blocks/transformer_encoder_layer.hpp>
#include <ecvm/models/vit/vit.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>

ViTOptions::ViTOptions() = default;

ViT::ViT(const ViTOptions &ops) {
  this->patch_embedding = PatchEmbedding(
      PatchEmbeddingOptions()
      .embed_dim(ops.embed_dim())
      .img_size(ops.img_size())
      .patch_size(ops.patch_size())
      .in_channels(ops.in_channels())
  );

  auto n_patches = this->patch_embedding->n_patches;

  this->position_embeddings = torch::zeros({1, n_patches, (int64_t)ops.embed_dim()}, torch::requires_grad());

  for (int i=0; i<ops.num_layers(); i++) {
    this->transformer_layers->push_back(TransformerEncoderLayer(
          TransformerEncoderLayerOptions()
          .embed_dim(ops.embed_dim())
          .num_heads(ops.num_heads())
          .mlp_dim(ops.mlp_dim())
          .dropout_rate(ops.dropout_rate())
    ));
  }

  this->flatten = torch::nn::Flatten();

  auto mlp_head_dim = ops.mlp_head_dim();
  auto mlp_head_layers = ops.mlp_head_layers();

  this->mlp_head = torch::nn::Sequential(
    torch::nn::Linear(ops.embed_dim() * n_patches, mlp_head_dim),
    torch::nn::BatchNorm1d(mlp_head_dim),
    torch::nn::ReLU(),
    torch::nn::Dropout(ops.dropout_rate())
  );

  int downscaling_factor = 1;
  for (auto i = 0; i + 2 < mlp_head_layers; i++) {
    torch::nn::Sequential blocks(
        torch::nn::Linear(mlp_head_dim / downscaling_factor, mlp_head_dim / (downscaling_factor * 2)),
        torch::nn::BatchNorm1d(mlp_head_dim / (downscaling_factor * 2)),
        torch::nn::ReLU(),
        torch::nn::Dropout(ops.dropout_rate())
    );
    for (const auto& block : *blocks) {
      this->mlp_head->push_back(block);
    }
    downscaling_factor *= 2;
  }

  this->mlp_head->push_back(torch::nn::Linear(mlp_head_dim / downscaling_factor, ops.num_classes()));

  register_module("patch_embedding", this->patch_embedding);
  register_parameter("position_embeddings", this->position_embeddings);
  register_module("transformer_layers", this->transformer_layers);
  register_module("mlp_head", this->mlp_head);
  register_module("flatten", this->flatten);
}

auto ViT::forward(torch::Tensor x) -> torch::Tensor {
  x = this->patch_embedding->forward(x);
  x = x + this->position_embeddings;
  x = this->transformer_layers->forward(x);
  x = this->flatten(x);
  x = this->mlp_head->forward(x);
  return x;
}
