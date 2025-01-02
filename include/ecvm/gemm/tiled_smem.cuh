#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void gemm_tiled_smem(float *c, float *a, float *b, int n, int k, int m) {
  const int crow = blockIdx.x * BLOCKSIZE;
  const int ccol = blockIdx.y * BLOCKSIZE;
  const int threadrow = threadIdx.x / BLOCKSIZE;
  const int threadcol = threadIdx.x % BLOCKSIZE;
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  a += crow * k;
  b += ccol;
  c += crow * m + ccol;

  float tmp = 0.0;

  for (int block_idx = 0; block_idx < k; block_idx += BLOCKSIZE) {
    As[threadrow * BLOCKSIZE + threadcol] = a[threadrow * k + threadcol];
    Bs[threadrow * BLOCKSIZE + threadcol] = b[threadrow * m + threadcol];

    __syncthreads();
    a += BLOCKSIZE;
    b += BLOCKSIZE * m; 

    for (int l=0; l < BLOCKSIZE; l++) {
      tmp += As[threadrow * BLOCKSIZE + l] * Bs[l * BLOCKSIZE + threadcol];
    }

    __syncthreads();
  }

  c[threadrow * m + threadcol] = tmp;
}
