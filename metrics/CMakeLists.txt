# metrics/CMakeLists.txt

# Create a library for metrics
add_library(metrics
    accuracy.cpp
)

# Link required libraries
target_link_libraries(metrics PUBLIC ecvm_core)

# Add include directories
target_include_directories(metrics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
