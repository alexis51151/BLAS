//
// Created by alg on 10/19/23.
//

#ifndef CUDA_BLAS_SAXPY_CUH
#define CUDA_BLAS_SAXPY_CUH

__global__
void saxpy_kernel(int n, float a, const float* x, const float* y, float* z) {
    // 2D -> 1D index
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    // saxpy computation
    if (id < n) {
        z[id] = a * x[id] + y[id];
    }
}

#endif //CUDA_BLAS_SAXPY_CUH
