#pragma once
#include <cuda_runtime.h>

__global__ void gemm_naive(float *c, float *a, float *b, int n, int k, int m);
