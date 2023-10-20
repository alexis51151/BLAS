#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

#include <cublas_v2.h>

#include "../utils.h" // File with macros to handle CUDA errors

using namespace std;

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

void bench_cublas(int n_trials) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    const float a = 2.0;

    // Allocate the vectors on host
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));

    // Initialize the vectors on host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 3.0;
    }

    // Define CUBLAS handle
    cublasHandle_t cublasH = NULL;

    // Initialize CUBLAS
    CUBLAS_CHECK(cublasCreate(&cublasH));

    // Allocate the vectors on device
    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    // Copy the vectors from host to devices
    CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    long avg_dur = 0;
    for (int i = 0; i < n_trials; i++) {
        CUDA_CHECK(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));
        auto start = chrono::high_resolution_clock::now();
        cublasSaxpy(cublasH, N, &a, d_x, 1, d_y, 1);
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

    // Free the CUBLAS handle
    CUBLAS_CHECK(cublasDestroy(cublasH));

    // Free the vectors on device
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Free the vectors on host
    free(x);
    free(y);
}


int main() {
    bench_cublas(10);
    return 0;
}
