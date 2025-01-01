#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void gemm_tiled(float *c, float *a, float *b, int n, int k, int m) {
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < n && y < m) {
    float tmp = 0.0;
    for (int i = 0; i < k; ++i) {
      tmp += a[x * k + i] * b[i * m + y];
    }
    c[x * m + y] = tmp;
  }
}
