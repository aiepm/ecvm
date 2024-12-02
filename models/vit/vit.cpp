#include <ecvm/blocks/patch_embedding.hpp>
#include <ecvm/blocks/transformer_encoder_layer.hpp>
#include <ecvm/models/vit/vit.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/dropout.h>

ViTOptions::ViTOptions() = default;

auto ViTOptions::ImgSize(int x) -> ViTOptions& {
  this->img_size = x;
  return *this;
}

auto ViTOptions::PatchSize(int x) -> ViTOptions& {
  this->patch_size = x;
  return *this;
}

auto ViTOptions::InChannels(int x) -> ViTOptions& {
  this->in_channels = x;
  return *this;
}

auto ViTOptions::NumClasses(int x) -> ViTOptions& {
  this->num_classes = x;
  return *this;
}

auto ViTOptions::EmbedDim(int x) -> ViTOptions& {
  this->embed_dim = x;
  return *this;
}

auto ViTOptions::NumHeads(int x) -> ViTOptions& {
  this->num_heads = x;
  return *this;
}

auto ViTOptions::MLPDim(int x) -> ViTOptions& {
  this->mlp_dim = x;
  return *this;
}

auto ViTOptions::NumLayers(int x) -> ViTOptions& {
  this->num_layers = x;
  return *this;
}

auto ViTOptions::MLPHeadDim(int x) -> ViTOptions& {
  this->mlp_head_dim = x;
  return *this;
}

auto ViTOptions::MLPHeadLayers(int x) -> ViTOptions& {
  this->mlp_head_layers = x;
  return *this;
}

auto ViTOptions::DropoutRate(double x) -> ViTOptions& {
  this->dropout_rate = x;
  return *this;
}

ViT::ViT(const ViTOptions &ops) {
  this->patch_embedding = PatchEmbedding(
      PatchEmbeddingOptions()
      .EmbedDim(ops.embed_dim)
      .ImgSize(ops.img_size)
      .PatchSize(ops.patch_size)
      .InChannels(ops.in_channels)
  );

  int n_patches = this->patch_embedding->n_patches;

  this->position_embeddings = torch::zeros({1, n_patches, ops.embed_dim}, torch::requires_grad());

  for (int i=0; i<ops.num_layers; i++) {
    this->transformer_layers->push_back(TransformerEncoderLayer(
          TransformerEncoderLayerOptions()
          .EmbedDim(ops.embed_dim)
          .NumHeads(ops.num_heads)
          .MLPDim(ops.mlp_dim)
          .DropoutRate(ops.dropout_rate)
    ));
  }

  this->flatten = torch::nn::Flatten();

  int mlp_head_dim = ops.mlp_head_dim;
  int mlp_head_layers = ops.mlp_head_layers;

  this->mlp_head = torch::nn::Sequential(
    torch::nn::Linear(ops.embed_dim * n_patches, mlp_head_dim),
    torch::nn::BatchNorm1d(mlp_head_dim),
    torch::nn::ReLU(),
    torch::nn::Dropout(ops.dropout_rate)
  );

  int downscaling_factor = 1;
  for (int i=0; i<mlp_head_layers - 2; i++) {
    torch::nn::Sequential blocks(
        torch::nn::Linear(mlp_head_dim / downscaling_factor, mlp_head_dim / (downscaling_factor * 2)),
        torch::nn::BatchNorm1d(mlp_head_dim / (downscaling_factor * 2)),
        torch::nn::ReLU(),
        torch::nn::Dropout(ops.dropout_rate)
    );
    for (const auto& block : *blocks) {
      this->mlp_head->push_back(block);
    }
    downscaling_factor *= 2;
  }

  this->mlp_head->push_back(torch::nn::Linear(mlp_head_dim / downscaling_factor, ops.num_classes));

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
