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
    src/kv_cache.cpp
    src/packed_model_reader.cpp
    src/packed_tensor_reader.cpp
    src/profiler.cpp
    src/reference_kernels.cpp
    src/tokenizer.cpp
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

add_executable(
    run_layer0_rmsnorm
    tools/run_layer0_rmsnorm.cpp
)

target_link_libraries(
    run_layer0_rmsnorm
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    test_rmsnorm
    tests/test_rmsnorm.cpp
)

target_link_libraries(
    test_rmsnorm
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    test_rope
    tests/test_rope.cpp
)

target_link_libraries(
    test_rope
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    test_matvec
    tests/test_matvec.cpp
)

target_link_libraries(
    test_matvec
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    run_tokenizer
    tools/run_tokenizer.cpp
)

target_link_libraries(
    run_tokenizer
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    dump_reference_tensors
    tools/dump_reference_tensors.cpp
)

target_link_libraries(
    dump_reference_tensors
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    dump_android_tensors
    tools/dump_android_tensors.cpp
)

target_link_libraries(
    dump_android_tensors
    PRIVATE
    ondevai_custom_runtime
)

add_executable(
    run_forward
    tools/run_forward.cpp
)

target_link_libraries(
    run_forward
    PRIVATE
    ondevai_custom_runtime
)
