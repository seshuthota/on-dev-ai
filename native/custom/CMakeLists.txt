add_library(
    ondevai_custom_runtime
    STATIC
    src/runtime.cpp
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
