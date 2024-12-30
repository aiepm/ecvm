add_library(device
  init.cu
)

target_link_libraries(device PUBLIC ecvm_core)

target_include_directories(device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
