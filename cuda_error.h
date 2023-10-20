//
// Created by alexis51151 on 06/02/23.
//

/*
 * Macros to check return value of calls to CUDA API
 * in case there was an error.
 */

// Source: https://leimao.github.io/blog/Proper-CUDA-Error-Checking/

#ifndef UTILS_CUDA_ERROR_H
#define UTILS_CUDA_ERROR_H
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
         std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
         std::exit(EXIT_FAILURE);
    }
}

#endif //UTILS_CUDA_ERROR_H
