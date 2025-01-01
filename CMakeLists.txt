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
set(CMAKE_CUDA_FLAGS_DEBUG "-O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -fmad=true")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

option(BUILD_TESTS "Build tests" OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(core)
add_subdirectory(device)
add_subdirectory(tensor)
add_subdirectory(gemm)

add_library(ecvm STATIC ecvm.cu)

target_link_libraries(ecvm PUBLIC ecvm_core device gemm tensor)

add_subdirectory(pipelines)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


