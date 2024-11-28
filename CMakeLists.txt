cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ecvm LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the global ecvm library
add_library(ecvm STATIC ecvm.cpp)

# Add the include directory to the global include path
target_include_directories(ecvm PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Link Torch to the ecvm library
target_link_libraries(ecvm PUBLIC "${TORCH_LIBRARIES}")

# Add subdirectories
add_subdirectory(blocks)
add_subdirectory(models)
# add_subdirectory(pipelines)
# add_subdirectory(tests)
