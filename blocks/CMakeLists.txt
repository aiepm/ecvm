# blocks/CMakeLists.txt

# Create a library for blocks
add_library(blocks
    patch_embedding.cpp
    transformer_encoder_layer.cpp
)

# Link required libraries
target_link_libraries(blocks PRIVATE ecvm)

# Add include directories
target_include_directories(blocks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
