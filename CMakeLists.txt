cmake_minimum_required(VERSION 3.14)
project(composable_kernel_project LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(HIP)

# Find HIP package
find_package(hip REQUIRED)

# Find Composable Kernel package
find_package(composable_kernel REQUIRED)

# Set HIP architectures
set(HIP_ARCHITECTURES "gfx908;gfx90a;gfx940;gfx941;gfx942")

# Define the extension target
add_library(fp8_gemm_lib SHARED
    csrc/fp8_gemm.cpp
    csrc/fp8_gemm.h
)

# Set target properties
set_target_properties(fp8_gemm_lib PROPERTIES
    LANGUAGE HIP
    LINKER_LANGUAGE HIP
)

# Add include directories
target_include_directories(fp8_gemm_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

# Link libraries
target_link_libraries(fp8_gemm_lib PRIVATE
    hip::device
    composable_kernel::composable_kernel
)

# Set compile options
target_compile_options(fp8_gemm_lib PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--amdgpu-target=${HIP_ARCHITECTURES}>
)

# Add executable
add_executable(main csrc/main.cpp)

# Link the executable with the library
target_link_libraries(main PRIVATE fp8_gemm_lib)