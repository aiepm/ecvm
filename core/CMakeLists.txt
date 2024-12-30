enable_language(CUDA)


add_library(ecvm_core INTERFACE)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS "/usr/local/cuda/lib64" "/usr/local/cuda/lib" REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "CORE OpenCV Libraries: ${OpenCV_LIBS}")

target_include_directories(ecvm_core INTERFACE ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ecvm_core INTERFACE ${OpenCV_LIBS} ${NVTOOLSEXT_LIBRARY} CUDA::cudart CUDA::cublas CUDA::cublasLt)
