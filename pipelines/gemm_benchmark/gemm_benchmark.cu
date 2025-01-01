#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <ecvm/gemm/naive.cuh>
#include <ecvm/gemm/tiled.cuh>
#include <ecvm/device/init.cuh>
#include <ecvm/tensor/init.cuh>
#include <ecvm/tensor/ops.cuh>
#include <nvtx3/nvToolsExt.h>
#include <string>
using i64 = int64_t;

constexpr float EPS = 1e-2;

auto get_time() -> i64 {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

auto get_microseconds(cudaEvent_t &start, cudaEvent_t &stop) -> i64 {
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  return ms * 1000;
}

auto matmul_naive_vs_cublas() -> int {
  nvtxRangePush("matmul");
  int n = 4096;
  int k = 4096;
  int m = 4096;

  int BLOCK_SIZE = 1024;
  int NUM_BLOCKS = (n * m + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *d_a, *d_b, *d_c_ref, *d_c;
  nvtxRangePush("matrix memory allocation");
  cudaMalloc(&d_a, n * k * sizeof(float));
  cudaMalloc(&d_b, k * m * sizeof(float));
  cudaMalloc(&d_c_ref, n * m * sizeof(float));
  cudaMalloc(&d_c, n * m * sizeof(float));
  nvtxRangePop();

  auto run = [&](int iter_num, std::string name, float eps) -> int {
    float alpha = 1.0, beta = 0.0;
    i64 total_matmul_time = 0, total_cublas_time = 0, total_coalesced_time = 0;

    curandState *s_a, *s_b;
    cudaMalloc(&s_a, n * k * sizeof(curandState));
    cudaMalloc(&s_b, k * m * sizeof(curandState));

    
    bool *d_res, h_res;
    cudaMalloc(&d_res, sizeof(bool));

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    for (int iter=0; iter<iter_num; iter++) {
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_a, time(nullptr), n, k);
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_b, time(nullptr), k, m);

      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, s_a, n, k);
      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_b, s_b, k, m);

      cudaDeviceSynchronize();

      cudaEvent_t coalesced_start, coalesced_end;
      cudaEventCreate(&coalesced_start);
      cudaEventCreate(&coalesced_end);
      // gridDim stays the same
      dim3 gridDim(4096 / 32, 4096 / 32);
      // make blockDim 1-dimensional, but don't change number of threads
      dim3 blockDim(32 * 32);
      nvtxRangePush("gemm_coalesced execution");
      cudaEventRecord(coalesced_start);
      gemm_tiled<32><<<gridDim, blockDim>>>(d_c, d_a, d_b, n, k, m);
      cudaEventRecord(coalesced_end);
      nvtxRangePop();

      cudaDeviceSynchronize();

      cudaEvent_t naive_start, naive_end;
      cudaEventCreate(&naive_start);
      cudaEventCreate(&naive_end);

      nvtxRangePush("gemm_naive execution");
      cudaEventRecord(naive_start);
      gemm_naive<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_c, d_a, d_b, n, k, m);
      cudaEventRecord(naive_end);
      nvtxRangePop();

      cudaDeviceSynchronize();
      
      cudaEvent_t cublas_start, cublas_end;
      cudaEventCreate(&cublas_start);
      cudaEventCreate(&cublas_end);

      nvtxRangePush("cublas sgemm execution");
      cudaEventRecord(cublas_start);
      cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_b, m, d_a, k, &beta, d_c_ref, m);
      cudaEventRecord(cublas_end);
      nvtxRangePop();

      cudaDeviceSynchronize();

      total_matmul_time += get_microseconds(naive_start, naive_end);
      total_cublas_time += get_microseconds(cublas_start, cublas_end);
      total_coalesced_time += get_microseconds(coalesced_start, coalesced_end);

      h_res = true;
      cudaMemcpy(d_res, &h_res, sizeof(bool), cudaMemcpyHostToDevice);
      check_matrix_equality_atomic<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_c, d_c_ref, n, m, d_res, eps);
      cudaDeviceSynchronize();
      cudaMemcpy(&h_res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);

      if (!h_res) {
        return 1;
      }
    }

    i64 average_matmul_time = total_matmul_time / iter_num;
    i64 average_cublas_time = total_cublas_time / iter_num;
    i64 average_coalesced_time = total_coalesced_time / iter_num;
    double average_matmul_flops = ((2.0 * n * m * k) / average_matmul_time) / 1e3;
    double average_coalesced_flops = ((2.0 * n * m * k) / average_coalesced_time) / 1e3;

    std::printf("%s avg naive time: %ld\n", name.c_str(), average_matmul_time);
    std::printf("%s avg naive gflops: %lf\n", name.c_str(), average_matmul_flops);
    std::printf("%s avg coalesced time: %ld\n", name.c_str(), average_coalesced_time);
    std::printf("%s avg coalesced gflops: %lf\n", name.c_str(), average_coalesced_flops);
    std::printf("%s avg cublas time: %ld\n", name.c_str(), average_cublas_time);

    cudaFree(s_a);
    cudaFree(s_b);
    cublasDestroy_v2(handle);
    return 0;
  };

  int warmup_runs = 10;
  int perf_runs = 100;

  if(auto err = run(warmup_runs, "warmup", EPS); err != 0) {
    std::printf("Warmup failed\n");
    return err;
  }
  std::printf("Warmup successfull, naive = cublas\n");

  if (auto err = run(perf_runs, "perf", EPS); err != 0) {
    std::printf("Perf failed\n");
    return err;
  } 
  std::printf("Perf successfull, naive = cublas\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_c_ref);
  return 0;
}

auto main() -> int {
  if (auto err = init_device(); err != 0) {
    return err;
  }
  if (auto err = matmul_naive_vs_cublas(); err != 0) {
    return err;
  }
  return 0;
}
