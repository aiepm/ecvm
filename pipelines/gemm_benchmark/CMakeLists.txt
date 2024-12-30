# pipelines/cifar-10-vit/CMakeLists.txt

# Create a library f
add_executable(gemm_benchmark
  gemm_benchmark.cu
)

target_link_libraries(gemm_benchmark PRIVATE ecvm)

# Add include directories
target_include_directories(gemm_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include) 

