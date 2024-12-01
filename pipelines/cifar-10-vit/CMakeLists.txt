# pipelines/cifar-10-vit/CMakeLists.txt

# Create a library f
add_executable(cifar-10-vit
  cifar-10-vit.cpp
)

target_link_libraries(cifar-10-vit PRIVATE ecvm)

# Add include directories
target_include_directories(cifar-10-vit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include) 

