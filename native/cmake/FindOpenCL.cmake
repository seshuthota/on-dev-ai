# Custom FindOpenCL for Android cross-compilation.
#
# Provides OpenCL_INCLUDE_DIRS (headers for compilation) and OpenCL_LIBRARY
# (ICD loader library for linking). The GGML_OPENCL backend uses dlopen/dlsym
# at runtime to invoke the actual OpenCL implementation.
#
# The vendor's libOpenCL.so (Qualcomm ICD loader) is bundled into the APK.
# It is accessed via <uses-native-library android:name="libOpenCL.so"/> in
# AndroidManifest.xml, which makes it available from the vendor namespace.
#
# Note: ggml-opencl uses ${OpenCL_INCLUDE_DIRS} (plural) for include dirs.

set(OpenCL_FOUND TRUE)
set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/OpenCL-Headers")
set(OpenCL_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../third_party/OpenCL-Headers" CACHE PATH "" FORCE)
set(OpenCL_LIBRARY "${CMAKE_CURRENT_LIST_DIR}/../third_party/libOpenCL.so" CACHE FILEPATH "" FORCE)
set(OpenCL_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../third_party/libOpenCL.so" CACHE FILEPATH "" FORCE)

add_library(OpenCL::OpenCL SHARED IMPORTED)
set_target_properties(OpenCL::OpenCL PROPERTIES
    IMPORTED_LOCATION "${OpenCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}"
)
