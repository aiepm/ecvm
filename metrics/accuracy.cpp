#include <ecvm/metrics/accuracy.hpp>

auto calculate_accuracy(const torch::Tensor& predictions, const torch::Tensor& labels) -> double {
    auto predicted_classes = predictions.argmax(1);

    auto correct = (predicted_classes == labels).sum().item<int64_t>();

    return static_cast<double>(correct) / labels.size(0); // Return accuracy as a fraction
}

