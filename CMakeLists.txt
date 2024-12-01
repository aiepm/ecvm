cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ecvm LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(core)
add_subdirectory(blocks)
add_subdirectory(datasets)
add_subdirectory(models)

add_library(ecvm STATIC ecvm.cpp)
target_link_libraries(ecvm PUBLIC ecvm_core blocks datasets models)

add_subdirectory(pipelines)

# add_subdirectory(tests)
