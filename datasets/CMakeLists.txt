add_library(datasets INTERFACE)

add_subdirectory(cifar-10)

target_link_libraries(datasets INTERFACE cifar-10)
