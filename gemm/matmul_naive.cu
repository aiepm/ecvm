#include <ecvm/gemm/matmul_naive.cuh>

__global__ void matmul_gpu(float *c, float *a, float *b, int n, int k, int m) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n * m) return;
  int row = id / m, col = id % m;
  float res = 0;
  for (int l=0; l<k; l++) {
    res += a[row * k + l] * b[l * m + col];
  }
  c[id] = res;
}
