#pragma once
#include <cuda_runtime.h>

__global__ void matmul_gpu(float *c, float *a, float *b, int n, int k, int m);

