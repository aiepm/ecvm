cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ecvm LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Specify the CUDA standard for device code (use C++17 for CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -fmad=true")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_subdirectory(core)
add_subdirectory(device)
add_subdirectory(tensor)
add_subdirectory(gemm)

add_library(ecvm STATIC ecvm.cu)

target_link_libraries(ecvm PUBLIC ecvm_core device gemm tensor)

add_subdirectory(pipelines)

# Debug configuration with no optimization
set(CMAKE_CUDA_FLAGS_DEBUG "-O0")

# Release configuration with high optimization
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# RelWithDebInfo configuration with moderate optimization
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2")
