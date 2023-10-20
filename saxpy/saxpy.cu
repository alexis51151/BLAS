#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

#include "saxpy.cuh"
#include "../cuda_error.h" // File with macros to handle CUDA errors

using namespace std;

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

// Sequential saxpy function on CPU in C
void saxpy(int n, float a, const float* x, const float* y, float* z) {
    for (int i = 0; i < n; i++) {
        z[i] = a * x[i] + y[i];
    }
}

// Sequential saxpy function on CPU in C++
void saxpy_cpp(float a, const vector<float>& x, const vector<float>& y, vector<float>& z) {
    for (int i = 0; i < z.size(); i++) {
        z[i] = a * x[i] + y[i];
    }
}


template<typename T>
void bench(int n_trials, T func_saxpy) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    float a = 2.0;

    // Allocate the vectors
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));
    float* z = (float*) malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 3.0;
        z[i] = 0.0;
    }

    long avg_dur = 0;
    for (int i = 0; i < n_trials; i++) {
        auto start = chrono::high_resolution_clock::now();
        func_saxpy(N, a, x, y, z);
        auto end = chrono::high_resolution_clock::now();
        long dur_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        avg_dur += dur_ms;
    }
    avg_dur = avg_dur / n_trials;
    cout << "[saxpy]: avg(duration) = " << avg_dur << " ms" << endl;


    // Check the output for the last call
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        assert(z[i] == 5);
        sum += z[i];
    }
    cout << "sum = " << sum << endl;
    // remark: sum becomes incorrect when N is too large
//    assert(sum == N * 5);

    // Free the vectors
    free(x);
    free(y);
    free(z);
}

template<typename T>
void bench_cpp(int n_trials, T func_saxpy) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    float a = 2.0;

    // Allocate the vectors
    vector<float> x;
    vector<float> y;
    vector<float> z;

    for (int i = 0; i < N; i++) {
        x.push_back(1.0);
        y.push_back(3.0);
        z.push_back(0.0);
    }

    long avg_dur = 0;
    for (int i = 0; i < n_trials; i++) {
        auto start = chrono::high_resolution_clock::now();
        func_saxpy(a, x, y, z);
        auto end = chrono::high_resolution_clock::now();
        long dur_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        avg_dur += dur_ms;
    }
    avg_dur = avg_dur / n_trials;
    cout << "[saxpy_cpp]: avg(duration) = " << avg_dur << " ms" << endl;


    // Check the output for the last call
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
//        assert(z[i] == 5);
        sum += z[i];
    }
    cout << "sum = " << sum << endl;
    // remark: sum becomes incorrect when N is too large
//    assert(sum == N * 5);
}

template<typename T>
void bench_cuda(int n_trials, T func_saxpy) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    float a = 2.0;

    // Allocate the vectors on host
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));
    float* z = (float*) malloc(N * sizeof(float));

    // Initialize the vectors on host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 3.0;
        z[i] = 0.0;
    }

    // Allocate the vectors on device
    float* d_x;
    float* d_y;
    float* d_z;
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_z, N * sizeof(float)));

    // Copy the vectors from host to devices
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));
    // No need to copy z to device, as its values will be overwritten

    // Compute the number of blocks
    int threads_per_block = 1024; // number of threads per block
    int n_blocks = (N + threads_per_block - 1) / threads_per_block; // ceil(N / threads_per_block)

    long avg_dur = 0;
    for (int i = 0; i < n_trials; i++) {
        auto start = chrono::high_resolution_clock::now();
        func_saxpy<<<n_blocks, threads_per_block>>>(N, a, d_x, d_y, d_z);
        CHECK_LAST_CUDA_ERROR();
        // Device synchronization barrier
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto end = chrono::high_resolution_clock::now();
        long dur_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        avg_dur += dur_ms;
    }
    avg_dur = avg_dur / n_trials;
    cout << "[saxpy_cuda]: avg(duration) = " << avg_dur << " ms" << endl;

    // Copy back the output from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Check the output for the last call
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        assert(z[i] == 5);
        sum += z[i];
    }
    cout << "sum = " << sum << endl;
    // remark: sum becomes incorrect when N is too large
//    assert(sum == N * 5);

    // Free the vectors on device
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_z));

    // Free the vectors on host
    free(x);
    free(y);
    free(z);
}


int main() {
//    bench(10, saxpy);
//    bench_cpp(10, saxpy_cpp);
    bench_cuda(10, saxpy_kernel);
    return 0;
}
