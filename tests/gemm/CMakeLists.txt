find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_gemm
  naive.cu
)

target_link_libraries(test_gemm PRIVATE
  ecvm
  GTest::gtest_main
)

# Register your tests with CTest
add_test(NAME test_gemm COMMAND test_gemm)
