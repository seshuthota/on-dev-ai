# Host toolchain for building vulkan-shaders-gen on Linux x86_64
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_MAKE_PROGRAM /home/curious/Android/Sdk/cmake/3.22.1/bin/ninja CACHE FILEPATH "" FORCE)

set(CMAKE_C_COMPILER /usr/bin/gcc CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE FILEPATH "" FORCE)
