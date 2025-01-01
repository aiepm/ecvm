add_library(gemm
  naive.cu
  tiled.cu
)

target_link_libraries(gemm PUBLIC ecvm_core)

target_include_directories(gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
