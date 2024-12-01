add_library(ecvm_core INTERFACE)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "CORE OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "CORE TORCH Libraries: ${TORCH_LIBRARIES}")  

target_include_directories(ecvm_core INTERFACE ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ecvm_core INTERFACE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
