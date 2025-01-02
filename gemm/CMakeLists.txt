add_library(gemm
  naive.cu
  tiled.cu
  tiled_smem.cu
)

target_link_libraries(gemm PUBLIC ecvm_core)

target_include_directories(gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
