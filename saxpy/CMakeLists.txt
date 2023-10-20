cmake_minimum_required(VERSION 3.26)
project(CUDA_BLAS LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89) # My GPU is RTX 4060 Ti (Compute Capability 8.9)

add_compile_options(-O3 -Wall -Wextra)

add_executable(CUDA_BLAS saxpy.cpp)
