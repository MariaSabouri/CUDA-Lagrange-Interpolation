# CUDA Lagrange Interpolation

This project implements **Lagrange interpolation** using two approaches:

-   Serial implementation on the CPU
-   Parallel implementation using CUDA on the GPU

The project compares the two implementations in terms of execution time,
numerical error, Speedup, and the effect of CUDA `block_size`.

## Project Overview

The project studies Lagrange interpolation in a heterogeneous-computing
setting by comparing the Serial CPU implementation with the CUDA
implementation.

The experiments use different values of:

``` text
slice_number = 8, 16, 32, 64
query_num    = 10,000
               100,000
               1,000,000
```

The CUDA implementation also evaluates different block sizes.

## Serial Implementation

The Serial implementation contains three nested loops:

1.  The outer loop corresponds to the query points.
2.  The middle loop performs the summation in the Lagrange equation.
3.  The inner loop calculates the Lagrange basis.

The total error over the query points is accumulated during the
calculation and the mean error is calculated afterwards.

For a fair comparison with CUDA, the variables are defined using `float`
in both implementations. Using different numerical types can introduce
different rounding errors and produce larger differences between Serial
and CUDA results.

## CUDA Implementation

The CUDA implementation uses two kernels:

``` text
lagrangeGPU
mean_error_calc
```

### `lagrangeGPU`

One CUDA block is assigned to each query point.

``` text
query point 0 -> block 0
query point 1 -> block 1
query point 2 -> block 2
...
```

For each Lagrange summation term, one thread is used. Each thread
calculates its Lagrange basis and stores its result in shared memory.

The threads are organized into warps. Partial results are reduced using
shared memory, and the result of each warp is stored in a shared array.
The warp results are then combined to obtain the final interpolation
result for the query point.

The thread with index zero stores the final result in the global result
arrays.

### `mean_error_calc`

After `lagrangeGPU` finishes, the mean error of the query points is
calculated.

This kernel follows a similar shared-memory reduction approach, but the
results from all blocks must finally be accumulated into one global
variable.

The final accumulated error is divided by the number of query points to
obtain the mean error.

## Shared Memory Reduction

The reduction function is:

``` text
Shared_mem_sum
```

Its logic follows a **Dissemination sum using shared memory**, with a
small modification.

At every reduction step, one half of the threads in a warp need to
update their values, while the remaining threads do not need to perform
an update. The lane index inside the warp is therefore used to determine
which threads participate in each step.

## Main Program

The main part of the CUDA program runs on the CPU and performs the
initial variable setup and kernel launches.

Because the GPU capability used in the experiment is greater than 2,
`cudaMallocManaged` is used for memory allocation.

CUDA execution time is measured using CUDA events.

The recorded results contain:

``` text
slice_number
block_size
query_num
Time Elapsed(us)
mean_error
version
```

## Numerical Error

The mean absolute error is used to compare the interpolation result with
the actual function value.

The error for each query point is calculated as the absolute difference
between the interpolation result and the actual value. The mean is then
calculated over all query points.

The Serial and CUDA implementations produce very similar mean-error
values in the experiments. Small differences can result from the
different order of floating-point operations in the parallel
implementation.

## Performance Comparison

Speedup is calculated as:

``` text
Speedup = T_CPU / T_GPU
```

where `T_CPU` is the Serial execution time and `T_GPU` is the CUDA
execution time.

The measured Speedup values are:

    slice_number   query_num   CUDA (us)   Serial (us)    Speedup
  -------------- ----------- ----------- ------------- ----------
               8      10,000    5.53E+02      2.09E+03   3.77E+00
               8     100,000    1.40E+03      2.76E+04   1.97E+01
               8   1,000,000    9.08E+03      2.34E+05   2.57E+01
              16      10,000    5.19E+02      7.96E+03   1.53E+01
              16     100,000    1.34E+03      8.17E+04   6.08E+01
              16   1,000,000    1.34E+04      7.78E+05   5.79E+01
              32      10,000    5.80E+02      2.82E+04   4.87E+01
              32     100,000    2.13E+03      3.03E+05   1.43E+02
              32   1,000,000    1.32E+04      2.97E+06   2.25E+02
              64      10,000    8.87E+02      1.32E+05   1.49E+02
              64     100,000    5.18E+03      1.24E+06   2.39E+02
              64   1,000,000    3.55E+04      1.26E+07   3.54E+02

The CUDA version is faster than the CPU version in all tested cases.

For `slice_number = 8`, Speedup increases from `3.77` for 10,000 query
points to `25.7` for 1,000,000 query points.

For `query_num = 1,000,000`, Speedup increases as `slice_number`
increases:

``` text
slice_number = 8   -> 25.7
slice_number = 16  -> 57.9
slice_number = 32  -> 225
slice_number = 64  -> 354
```

For small problem sizes, overheads such as kernel launching, thread
management, and memory management represent a larger part of the
execution time. As the problem size increases, these overheads become
less significant compared with the computational workload, allowing the
GPU's parallel processing capability to be used more effectively.

## Effect of Block Size

The effect of CUDA block size is evaluated using:

``` text
block_size = 64
block_size = 128
block_size = 256
block_size = 512
```

The experiments are performed for:

``` text
query_num = 10,000
query_num = 100,000
query_num = 1,000,000
```

The results show that changing `block_size` has a significant effect on
CUDA execution time.

In most tested cases, increasing `block_size` from 64 to 128, 256, and
512 increases the execution time. However, the best block size is not
necessarily the same for every individual case.

For example, for `slice_number = 64` and `query_num = 1,000,000`:

    block_size   Time Elapsed (us)
  ------------ -------------------
            64            3.48E+04
           128            3.89E+04
           256            1.00E+05
           512            1.93E+05

Increasing the number of threads per block does not necessarily increase
performance. Larger blocks can consume more GPU resources such as
registers and shared memory, which may reduce the number of blocks that
can execute concurrently on an SM.

Therefore, selecting a suitable block size requires a balance between
parallelism and GPU resource usage.

## Conclusion

The experiments show that the CUDA implementation provides a significant
performance improvement over the Serial CPU implementation for the
tested cases.

The performance advantage becomes larger as the number of query points
and interpolation points increases because the computational workload
becomes larger relative to GPU execution overhead.

The experiments also show that block size has an important effect on
CUDA performance. Increasing the number of threads per block does not
necessarily improve performance because larger blocks can consume more
GPU resources and reduce the number of blocks that can execute
concurrently.

Therefore, both problem size and CUDA block size have an important role
in the performance of the implementation.

## Repository

The project source code is available at:

https://github.com/MariaSabouri/CUDA-Lagrange-Interpolation
