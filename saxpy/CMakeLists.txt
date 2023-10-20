cmake_minimum_required(VERSION 3.22)
project(CUDA_BLAS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89) # My GPU is RTX 4060 Ti (Compute Capability 8.9)

add_compile_options(-O3) # Compile flags for NVCC

# C version
add_executable(saxpy saxpy.c)
# C++ version
add_executable(saxpy_cpp saxpy.cpp)
# CUDA version
add_executable(saxpy_cuda saxpy.cu)
# CUBLAS version
add_executable(saxpy_cublas saxpy_cublas.cu)

# Link to CUBLAS for saxpy_cublas target
target_link_libraries(saxpy_cublas PUBLIC cublas)