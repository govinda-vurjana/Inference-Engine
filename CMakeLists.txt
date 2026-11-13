cmake_minimum_required(VERSION 3.20)

project(inference_engine
    VERSION 0.1.0
    DESCRIPTION "Inference Engine"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd / editor tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- library ----------------------------------------------------------
add_library(greeter
    src/greeter.cpp)

target_include_directories(greeter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- executable -------------------------------------------------------
add_executable(inference_engine
    src/main.cpp)

target_link_libraries(inference_engine PRIVATE greeter)

# ---- tests ------------------------------------------------------------
include(CTest)

if(BUILD_TESTING)
    add_executable(test_greeter tests/test_greeter.cpp)
    target_link_libraries(test_greeter PRIVATE greeter)
    add_test(NAME greeter.hello COMMAND test_greeter)
endif()
