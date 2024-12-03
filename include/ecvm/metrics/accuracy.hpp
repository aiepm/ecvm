#ifndef ACCURACY_HPP
#define ACCURACY_HPP

#include <torch/torch.h>

auto calculate_accuracy(const torch::Tensor& predictions, const torch::Tensor& labels) -> double;

#endif // !ACCURACY_HPP


