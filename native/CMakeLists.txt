cmake_minimum_required(VERSION 3.22.1)

project(ondevai_native LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(ONDEVAI_ENABLE_QNN "Enable experimental QNN integration plumbing" OFF)

if(ONDEVAI_ENABLE_QNN)
    add_compile_definitions(ONDEVAI_ENABLE_QNN=1)
else()
    add_compile_definitions(ONDEVAI_ENABLE_QNN=0)
endif()

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.5-a+fp16+i8mm+dotprod" CACHE STRING "" FORCE)
# GGML_PROFILER: compile-time flag to enable per-op profiling in ggml-cpu.c
# Runtime also requires GGML_PROFILER=1 env var
# Note: no FORCE so -DGGML_PROFILER=ON from gradle takes precedence
set(GGML_PROFILER OFF CACHE BOOL "")
if(GGML_PROFILER)
    add_compile_definitions(GGML_PROFILER)
endif()
set(GGML_OPENCL ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL_TARGET_VERSION "300" CACHE STRING "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_compile_definitions(GGML_OPENCL_DISABLE_FLASH_ATTN_KERNELS)

# --- QNN SDK discovery (when enabled) ---
if(ONDEVAI_ENABLE_QNN)
    # Discover QAIRT SDK include path. Gradle passes -DQAIRT_SDK_ROOT=... when QNN is enabled.
    if(DEFINED QAIRT_SDK_ROOT AND EXISTS "${QAIRT_SDK_ROOT}/include/QNN")
        set(QAIRT_SDK_INCLUDE_DIR "${QAIRT_SDK_ROOT}/include/QNN")
        message(STATUS "QAIRT SDK include: ${QAIRT_SDK_INCLUDE_DIR}")
    else()
        # Fallback: well-known repo-local path
        set(_QAIRT_FALLBACK "${CMAKE_CURRENT_LIST_DIR}/../v2.45.0.260326/qairt/2.45.0.260326")
        if(EXISTS "${_QAIRT_FALLBACK}/include/QNN")
            set(QAIRT_SDK_INCLUDE_DIR "${_QAIRT_FALLBACK}/include/QNN")
            message(STATUS "QAIRT SDK include (fallback): ${QAIRT_SDK_INCLUDE_DIR}")
        else()
            message(WARNING "ONDEVAI_ENABLE_QNN is ON but QAIRT SDK include dir not found. QNN runner will not be built.")
        endif()
    endif()

    if(DEFINED QAIRT_SDK_INCLUDE_DIR)
        add_subdirectory(qnn build-qnn EXCLUDE_FROM_ALL)
    endif()
endif()

add_subdirectory(custom build-custom)

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/llama.cpp)

add_subdirectory(${LLAMA_CPP_DIR} build-llama EXCLUDE_FROM_ALL)

add_library(
    ondevai_native
    SHARED
    jni/jni_bridge.cpp
    engine/src/app_engine.cpp
)

target_include_directories(
    ondevai_native
    PRIVATE
    engine/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

# When QNN runner is available, add its include path so jni_bridge can call it.
if(ONDEVAI_ENABLE_QNN AND TARGET qnn_runner)
    target_include_directories(
        ondevai_native
        PRIVATE
        qnn/include
    )
endif()

target_compile_features(ondevai_native PRIVATE cxx_std_20)

find_library(log-lib log)

target_link_libraries(
    ondevai_native
    llama
    ${log-lib}
)

# Link QNN runner into the app shared lib when available.
if(ONDEVAI_ENABLE_QNN AND TARGET qnn_runner)
    target_link_libraries(ondevai_native qnn_runner)
endif()
