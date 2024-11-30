add_library(ecvm_core STATIC ecvm_core.cpp)

target_include_directories(ecvm_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(ecvm_core INTERFACE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
