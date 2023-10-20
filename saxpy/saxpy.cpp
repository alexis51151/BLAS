#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

using namespace std;

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

// Sequential saxpy function on CPU in C++
void saxpy_cpp(float a, const vector<float>& x, const vector<float>& y, vector<float>& z) {
    for (int i = 0; i < z.size(); i++) {
        z[i] = a * x[i] + y[i];
    }
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
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
//        assert(z[i] == 5);
        sum += z[i];
    }
    cout << "sum = " << sum << endl;
    assert(sum == N * 5);
}


int main() {
    bench_cpp(10, saxpy_cpp);
    return 0;
}
