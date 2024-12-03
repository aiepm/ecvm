#include <ecvm/datasets/cifar-10/cifar-10.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <torch/types.h>

CIFAR10::CIFAR10(const std::string& image_dir, const std::string& label_file) {
  std::ifstream file(label_file);
  std::string img_name;
  int label;
  while (file >> img_name >> label) {
    img_name = image_dir + img_name;
    entries.emplace_back(img_name, label);
  }
}

// Get a single example (image, label)
auto CIFAR10::get(size_t index) -> torch::data::Example<> {
  auto &[path, label] = entries[index];
  cv::Mat img = cv::imread(path);
  if (img.empty()) {
    throw std::runtime_error("Failed to read image: " + path);
  }

  // Convert to tensor
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // Convert to RGB
  img.convertTo(img, CV_32F, 1.0 / 255.0);   // Normalize to [0, 1]
  auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
  tensor_img = tensor_img.permute({2, 0, 1}); // Change to CHW format

  auto labels_tensor = torch::tensor(label, torch::kInt64);

  return {tensor_img.clone(), labels_tensor.clone()};
}

// Get the size of the dataset
auto CIFAR10::size() const -> torch::optional<size_t> {
  return entries.size();
}

