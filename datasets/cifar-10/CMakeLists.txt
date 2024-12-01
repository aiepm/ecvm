add_library(cifar-10 STATIC cifar-10.cpp)

target_link_libraries(cifar-10 PUBLIC ecvm_core)

target_include_directories(cifar-10 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
