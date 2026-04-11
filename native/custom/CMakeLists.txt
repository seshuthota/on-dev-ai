if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)

    project(ondevai_custom_runtime CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
    ondevai_custom_runtime
    STATIC
    src/runtime.cpp
    src/packed_model_reader.cpp
)

target_include_directories(
    ondevai_custom_runtime
    PUBLIC
    include
)

target_compile_features(ondevai_custom_runtime PUBLIC cxx_std_20)

add_executable(
    custom_runtime_smoke
    tests/custom_runtime_smoke.cpp
)

target_link_libraries(
    custom_runtime_smoke
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    run_reference
    tools/run_reference.cpp
)

target_link_libraries(
    run_reference
    PRIVATE
    ondevai_custom_runtime
)
