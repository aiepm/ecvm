#pragma once
#include <curand_kernel.h>

__global__ auto initCurandStates(curandState *states, unsigned long seed, int rows, int cols) -> void;

__global__ auto generateRandomMatrix(float *matrix, curandState *states, int rows, int cols) -> void;
