#include <c10/core/TensorOptions.h>
#include <iostream>
#include <torch/torch.h>
#include <ecvm/models/vit/vit.hpp>
#include <torchvision/vision.h>

auto main() -> int {
  torch::DeviceType device_type;
  device_type = torch::kCUDA;
  return 0;
}
