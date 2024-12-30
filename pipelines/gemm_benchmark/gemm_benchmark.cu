#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <ecvm/gemm/matmul_naive.cuh>
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

auto matmul_cpu_vs_gpu() -> int {
  nvtxRangePush("matmul");
  int n = 4092;
  int k = 4092;
  int m = 4092;

  int BLOCK_SIZE = 1024;
  int NUM_BLOCKS = (n * m + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *d_a, *d_b, *d_c_ref, *d_c;
  nvtxRangePush("matrix memory allocation");
  cudaMalloc(&d_a, n * k * sizeof(float));
  cudaMalloc(&d_b, k * m * sizeof(float));
  cudaMalloc(&d_c_ref, n * m * sizeof(float));
  cudaMalloc(&d_c, n * m * sizeof(float));
  nvtxRangePop();

  auto fmatmul_naive = [&](i64 &timer, cudaStream_t &stream) -> int {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    nvtxRangePush("matmul_naive execution");
    cudaEventRecord(start, stream);
    matmul_gpu<<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>(d_c, d_a, d_b, n, k, m);
    cudaEventRecord(stop, stream);
    nvtxRangePop();
    cudaEventSynchronize(stop);
    timer += get_microseconds(start, stop);
    return 0;
  };

  auto fmatmul_cublas = [&](i64 &timer, cudaStream_t &stream) -> int {
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, stream);

    float alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvtxRangePush("cublas sgemm execution");
    cudaEventRecord(start, stream);
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_b, m, d_a, k, &beta, d_c_ref, m);
    cudaEventRecord(stop, stream);
    nvtxRangePop();
    cudaEventSynchronize(stop);

    timer += get_microseconds(start, stop);
    cublasDestroy_v2(handle);
    return 0;
  };

  auto run = [&](int iter_num, std::string name, float eps) -> int {
    i64 total_matmul_time = 0;
    i64 total_cublas_time = 0;

    curandState *s_a, *s_b;
    cudaMalloc(&s_a, n * k * sizeof(curandState));
    cudaMalloc(&s_b, k * m * sizeof(curandState));

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    for (int iter=0; iter<iter_num; iter++) {
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_a, time(nullptr), n, k);
      initCurandStates<<<NUM_BLOCKS, BLOCK_SIZE>>>(s_b, time(nullptr), k, m);

      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, s_a, n, k);
      generateRandomMatrix<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_b, s_b, k, m);

      fmatmul_naive(total_matmul_time, s1);
      fmatmul_cublas(total_cublas_time, s2);

      cudaStreamSynchronize(s1);
      cudaStreamSynchronize(s2);

      bool *d_res, h_res;
      h_res = true;
      cudaMalloc(&d_res, sizeof(bool));
      cudaMemcpy(d_res, &h_res, sizeof(bool), cudaMemcpyHostToDevice);
      check_matrix_equality_atomic<<<NUM_BLOCKS, BLOCK_SIZE, 0, s1>>>(d_c, d_c_ref, n, m, d_res, eps);
      cudaStreamSynchronize(s1);
      cudaMemcpy(&h_res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);

      if (!h_res) {
        return 1;
      }
    }
    
    i64 average_matmul_time = total_matmul_time / iter_num;
    i64 average_cublas_time = total_cublas_time / iter_num;

    std::printf("%s avg matmul time: %ld\n", name.c_str(), average_matmul_time);
    std::printf("%s avg cublas time: %ld\n", name.c_str(), average_cublas_time);

    cudaFree(s_a);
    cudaFree(s_b);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
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
  if (auto err = matmul_cpu_vs_gpu(); err != 0) {
    return err;
  }
  return 0;
}
