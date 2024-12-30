add_library(tensor
  init.cu 
  ops.cu 
)

target_link_libraries(tensor PUBLIC ecvm_core)

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
