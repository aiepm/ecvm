# datasets/cifar-10/CMakeLists.txt

add_library(cifar-10
    cifar-10.cpp
)

target_link_libraries(cifar-10 PRIVATE ecvm)

target_include_directories(cifar-10 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
