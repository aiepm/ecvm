cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ecvm LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the global ecvm library
add_library(ecvm STATIC ecvm.cpp)

# Add subdirectories
add_subdirectory(blocks)
add_subdirectory(core)
add_subdirectory(datasets)
add_subdirectory(models)
add_subdirectory(pipelines)

target_include_directories(ecvm PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ecvm PUBLIC ecvm_core datasets models)

include_directories(${OpenCV_INCLUDE_DIRS})

# add_subdirectory(tests)
