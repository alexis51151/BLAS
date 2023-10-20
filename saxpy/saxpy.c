#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>

/*
 * SAXPY stands for Single Precision A * X + Y where X,Y are vectors of size N.
 * Computation: Let a \in R, \forall i \in [1,N],
 * z_i = a * x_i + y_i
 */

// Sequential saxpy function on CPU in C
void saxpy(int n, float a, const float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void reset(int n, float* y, float val) {
    for (int i =0; i < n; i++) {
        y[i] = val;
    }
}

void bench(int n_trials, void* func_saxpy(int, float, const float*, float*)) {
    // Define the example: 2 * I + 3I
    const int N = 2 << 27;
    float a = 2.0;

    // Allocate the vectors
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 3.0;
    }

    long avg_dur = 0;
    struct timespec start, end;
    for (int i = 0; i < n_trials; i++) {
        reset(N, y, 3.0);
        clock_gettime(CLOCK_MONOTONIC, &start);
        func_saxpy(N, a, x, y);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double dur_ms = ((end.tv_sec - start.tv_sec) * 1e9 + end.tv_nsec - start.tv_nsec) / 1e6;
        avg_dur += dur_ms;
    }
    avg_dur = avg_dur / n_trials;
    printf("[saxpy] avg(duration) = %ld ms\n", avg_dur);

    // Check the output for the last call
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        assert(y[i] == 5.0);
        sum += y[i];
    }
    printf("sum = %e\n", sum);
    // remark: sum becomes incorrect when N is too large
    assert(sum == N * 5);

    // Free the vectors
    free(x);
    free(y);
}

int main() {
    bench(10, (void*) saxpy);
    return 0;
}
