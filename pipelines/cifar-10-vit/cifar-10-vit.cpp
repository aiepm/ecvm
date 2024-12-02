#include <iostream>
#include <torch/data/dataloader.h>
#include <torch/data/samplers/random.h>
#include <torch/data/samplers/sequential.h>
#include <torch/nn/init.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <ecvm/models/vit/vit.hpp>
#include <ecvm/datasets/cifar-10/cifar-10.hpp>
#include <torchvision/vision.h>

struct CONFIG {
  const int BATCH_SIZE = 4096;
  const int EPOCHS = 1000;
  const int NUM_WORKERS = 16;
};

auto main() -> int {
  CONFIG conf;

  torch::DeviceType device_type;
  device_type = torch::kCUDA;

  torch::Device device(device_type);

  std::string train_image_dir = "/core/datasets/cifar10/train/images/";
  std::string train_labels = "/core/datasets/cifar10/train/labels.txt";

  auto train_dataset = CIFAR10(train_image_dir, train_labels)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

  std::string test_image_dir = "/core/datasets/cifar10/test/images/";
  std::string test_labels = "/core/datasets/cifar10/test/labels.txt";

  auto test_dataset = CIFAR10(test_image_dir, test_labels)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

  auto model = std::make_shared<ViT>(ViTOptions());
  for (auto &param : model->parameters()) {
    torch::nn::init::uniform_(param);
  }
  model->to(device);

  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(conf.BATCH_SIZE).workers(conf.NUM_WORKERS).enforce_ordering(false));
  auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(conf.BATCH_SIZE).workers(conf.NUM_WORKERS).enforce_ordering(false));

  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).lr(0.01).momentum(0.9).nesterov(true).weight_decay(1e-4));

  auto train_step = [&](size_t epoch) {
    model->train();
    double running_loss = 0, cnt = 0;
    for (auto& batch : *train_data_loader) {
      auto data = batch.data.to(device), labels = batch.target.to(device);
      
      optimizer.zero_grad();

      auto output = model->forward(data);
      auto loss = torch::binary_cross_entropy_with_logits(output, labels);
      
      loss.backward();
      optimizer.step();

      running_loss += loss.item().toFloat();
      cnt += batch.data.size(0);
    }
    auto average_loss = running_loss / cnt;
    std::printf("Epoch %lu train loss %lf\n", epoch, average_loss);
  };

  auto eval_step = [&](size_t epoch) {
    model->eval();
    double running_loss = 0, cnt = 0;
    for (auto& batch : *test_data_loader) {
      auto data = batch.data.to(device), labels = batch.target.to(device);
      
      auto output = model->forward(data);
      auto loss = torch::binary_cross_entropy_with_logits(output, labels);
      
      running_loss += loss.item().toFloat();
      cnt += batch.data.size(0);
    }
    optimizer.zero_grad();
    auto average_loss = running_loss / cnt;
    std::printf("Epoch %lu test loss %lf\n", epoch, average_loss);
  };

  for (size_t epoch = 1; epoch <= conf.EPOCHS; epoch++) {
    train_step(epoch);
    eval_step(epoch);
  }

  return 0;
}
