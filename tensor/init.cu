#include <ecvm/tensor/init.cuh>

__global__ auto initCurandStates(curandState *states, unsigned long seed, int rows, int cols) -> void {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalElements = rows * cols;
  if (idx < totalElements) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

__global__ auto generateRandomMatrix(float *matrix, curandState *states, int rows, int cols) -> void {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalElements = rows * cols;
  if (idx < totalElements) {
    matrix[idx] = curand_uniform(&states[idx]);
  }
}
