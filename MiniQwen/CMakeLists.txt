cmake_minimum_required(VERSION 3.20)

project(MiniQwen
    VERSION 0.1.0
    LANGUAGES CXX
)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main tensor library
add_library(tensor_lib
    src/tensor.cpp
)

# Expose the include directory
target_include_directories(tensor_lib
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Main executable
add_executable(miniqwen
    src/main.cpp
)

target_link_libraries(miniqwen
    PRIVATE tensor_lib
)

# Test executable
add_executable(tensor_tests
    tests/tensor_tests.cpp
)

target_link_libraries(tensor_tests
    PRIVATE tensor_lib
)

# Enable CTest
enable_testing()

add_test(
    NAME TensorTests
    COMMAND tensor_tests
)