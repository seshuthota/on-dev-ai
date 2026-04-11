cmake_minimum_required(VERSION 3.22.1)

project(ondevai_qnn LANGUAGES CXX)

# Standalone smoke binary (built by scripts/build_qnn_smoke.sh, not part of the app)
add_executable(
    qnn_smoke
    src/qnn_smoke.cpp
)

target_compile_features(qnn_smoke PRIVATE cxx_std_17)

target_link_libraries(
    qnn_smoke
    dl
)

# In-process QNN runner library (linked into the app's native shared lib)
if(ONDEVAI_ENABLE_QNN AND DEFINED QAIRT_SDK_INCLUDE_DIR)
    add_library(
        qnn_runner
        STATIC
        src/qnn_runner.cpp
    )

    target_include_directories(
        qnn_runner
        PUBLIC
        include
        PRIVATE
        ${QAIRT_SDK_INCLUDE_DIR}
    )

    target_compile_features(qnn_runner PRIVATE cxx_std_17)

    find_library(log-lib-qnn log)

    target_link_libraries(
        qnn_runner
        dl
        ${log-lib-qnn}
    )
endif()
