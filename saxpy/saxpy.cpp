#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

#include "saxpy.cuh"

using namespace std;

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

struct BenchRes {
    long dur_ms;
    float* output;
};

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

int main() {
    bench(10, saxpy);
    bench_cpp(10, saxpy_cpp);
    return 0;
}
