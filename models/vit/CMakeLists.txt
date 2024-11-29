# models/vit/CMakeLists.txt

# Create a library for vit
add_library(vit
    vit.cpp
)

# Link required libraries
target_link_libraries(vit PRIVATE ecvm)

# Add include directories
target_include_directories(vit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

