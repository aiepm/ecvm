#ifndef CIFAR10_HPP
#define CIFAR10_HPP

#include <torch/torch.h>

struct CIFAR10 : torch::data::Dataset<CIFAR10> {
  std::vector<std::pair<std::string, int>> entries;

  CIFAR10(const std::string& image_dir, const std::string& label_file);

  // Get a single example (image, label)
  auto get(size_t index) -> torch::data::Example<> override;

  // Get the size of the dataset
  [[nodiscard]]
  auto size() const -> torch::optional<size_t> override;
};

#endif // !f CIFAR10_HPP

