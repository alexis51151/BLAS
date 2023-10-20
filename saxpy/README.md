# SAXPY


We consider the following benchmark settings:
- vector size: N = `2 << 27` (to have duration of about 100ms)
- computation: Z = 2.0 * I + 3I (I = identity vector of size N)
- 10 trials (we report the average duration)

### Benchmarking results :

| variant         | duration (ms) | sum         |
|-----------------|---------------|-------------|
| saxpy           | 79            | 1.34218e+09 |
| saxpy_cpp       | 78            | 1.34218e+09 |
| saxpy_cuda_128  | 13            | 1.34218e+09 |
| saxpy_cuda_1024 | 12            | 1.34218e+09 |
| saxpy_cublas    | 13            | 1.34218e+09 |


### Conclusion:

1. **C vs C++:** the C++ and C version perform similarly.
2. **CUDA: 128 vs 1024:** We observe that, for the CUDA variant, the number of threads per block does not impact the performance.
3. **CUDA vs C/C++:** CUDA parallelization allows for a **x10** speedup here, which is OK but not incredible.
4. **CUDA vs CUBLAS:** Similar performance, there may not be optimization opportunities here.