#include <iostream>
#include <chrono>
#include <cassert>

#include "../utils.h" // File with macros to handle CUDA errors

using namespace std;

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

__global__
void saxpy_kernel(int n, float a, const float* x, float* y) {
    // 2D -> 1D index
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    // saxpy computation
    if (id < n) {
        y[id] = a * x[id] + y[id];
    }
}

template<typename T>
void bench_cuda(int n_trials, T func_saxpy) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    float a = 2.0;

    // Allocate the vectors on host
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));

    // Initialize the vectors on host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 3.0;
    }

    // Allocate the vectors on device
    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    // Copy the vectors from host to devices
    CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    // Compute the number of blocks
    int threads_per_block = 128; // number of threads per block
    int n_blocks = (N + threads_per_block - 1) / threads_per_block; // ceil(N / threads_per_block)

    long avg_dur = 0;
    for (int i = 0; i < n_trials; i++) {
        CUDA_CHECK(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));
        auto start = chrono::high_resolution_clock::now();
        func_saxpy<<<n_blocks, threads_per_block>>>(N, a, d_x, d_y);
        // Device synchronization barrier
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = chrono::high_resolution_clock::now();
        long dur_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        avg_dur += dur_ms;
    }
    avg_dur = avg_dur / n_trials;
    cout << "[saxpy_cuda]: avg(duration) = " << avg_dur << " ms" << endl;

    // Copy back the output from device to host
    CUDA_CHECK(cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check the output for the last call
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        assert(y[i] == 5);
        sum += y[i];
    }
    cout << "sum = " << sum << endl;
    assert(sum == N * 5);

    // Free the vectors on device
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Free the vectors on host
    free(x);
    free(y);
}


int main() {
    bench_cuda(10, saxpy_kernel);
    return 0;
}
