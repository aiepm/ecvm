#include <iostream>
#include <torch/torch.h>
#include <ecvm/models/vit/vit.hpp>
#include <ecvm/datasets/cifar-10/cifar-10.hpp>
#include <torchvision/vision.h>

auto main() -> int {
  torch::DeviceType device_type;
  device_type = torch::kCUDA;

  std::string train_image_dir = "/core/datasets/cifar10/train/images";
  std::string train_labels = "/core/datasets/cifar10/train/labels.txt";

  auto train_dataset = CIFAR10(train_image_dir, train_labels);

  std::string test_image_dir = "/core/datasets/cifar10/test/images";
  std::string test_labels = "/core/datasets/cifar10/test/labels.txt";

  auto test_dataset = CIFAR10(test_image_dir, test_labels);

  return 0;
}
