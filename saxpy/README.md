# SAXPY


We consider the following benchmark settings:
- vector size: N = `2 << 27` (to have duration of about 100ms)
- computation: Z = 2.0 * I + 3I (I = identity vector of size N)
- 10 trials (we report the average duration)

### Benchmarking results :

| variant         | duration (ms) | sum         |
|-----------------|---------------|-------------|
| saxpy           | 108           | 1.34218e+08 |
| saxpy_cpp       | 94            | 1.34218e+08 |
| saxpy_cuda_128  | 12            | 1.34218e+08           |
| saxpy_cuda_1024 | 12            | 1.34218e+08           |


### Conclusion:

1. **C vs C++:** the C++ version seems to be somewhat more performant. May have to do with simpler allocation/deallocation.
2. **CUDA: 128 vs 1024:** We observe that, for the CUDA variant, the number of threads per block does not impact the performance.
3. **CUDA vs C/C++:** CUDA parallelization allows for a **x10** speedup here, which is OK but not incredible.