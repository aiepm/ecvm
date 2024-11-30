add_library(models INTERFACE)

add_subdirectory(vit)

target_link_libraries(models INTERFACE ecvm_core vit)
