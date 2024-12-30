#include <cstdio>
#include <ecvm/device/init.cuh>

auto init_device() -> int {
  int deviceCount;
  auto err = cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::printf("No cuda-compatible devices found.\n");
    return 1;
  }

  std::printf("Found %d device(s)\n", deviceCount);

  // Optionally print device properties
  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::printf("Device %d: %s\n", i, prop.name);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    std::printf("Failed to set device: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
