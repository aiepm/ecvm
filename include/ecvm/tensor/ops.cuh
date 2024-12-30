#pragma once

__global__ void check_matrix_equality_atomic(const float* A, const float* B, int n, int m, bool* result, float epsilon);
