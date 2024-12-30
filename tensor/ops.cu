#include <ecvm/tensor/ops.cuh>

__global__ void check_matrix_equality_atomic(const float* A, const float* B, int n, int m, bool* result, float epsilon) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n * m) return;

  if (fabsf(A[idx] - B[idx]) > epsilon) {
    atomicExch((int*)result, 0);
  }
}
