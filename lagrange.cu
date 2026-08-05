#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#define _USE_MATH_DEFINES
#include <math.h>
/*
nvcc lagrange.cu -o lagrange
./lagrange
*/

#define MAX_BLKSZ 1024
#define WARPSZ 32
#define BLOCK_COUNT 32

__device__ float Shared_mem_sum(float* shared_vals)
{
    int my_lane = threadIdx.x % warpSize;

    for(int diff == warpSize / 2; diff > 0; diff = diff / 2)
    {
        int source = (my_lane + diff) % warpSize;
        shared_vals[my_lane] += shared_vals[source];
    }
    return shared_vals[my_lane];
}

__device__ float U(float x)
{
    return sin(x);
}

__global__ void lagrangeGPU(
    const float a,
    const float b,
    const float h,
    const float x_input,
    const int n,
    float* lag,
    float* d_lag_based_arr_p
)
{
    __shared__ float thread_calcs[MAX_BLKSZ];
    __shared__ float warp_sum_arr[WARPSZ];

    int my_i = blockDim.x * blockIdx.x + threadIdx.x;

    if(my_i < n)
    {
        float my_x = a + my_i * h;
        float li = 1.0f;
        for (int k = 1; k < n; k++) {
            if (k != my_i) {
                float x_k = a + k * h;
                li *= (x_input - x_k) / (my_x - x_k);
            }        
        }
        d_lag_based_arr_p[my_i] = li;
    }
    // printf("Hello from GPU! Block %d Thread %d Lagrange_based_calculated %f \n",
    //        blockIdx.x, threadIdx.x, d_lag_based_arr_p[my_i]);                                                          
    __syncthreads();

    int my_warp = threadIdx.x / WARPSZ;
    int my_lane = threadIdx.x % WARPSZ;

    float* shared_vals = thread_calcs + my_warp * WARPSZ;
    float blk_result = 0.0;

    shared_vals[my_lane] = 0.0f;
    
    if(my_i < n)
    {
        float my_x = a + my_i * h;
        float my_y = U(my_x);
        shared_vals[my_lane] = my_y * d_lag_based_arr_p[my_i];
    }
    float my_result = Shared_mem_sum(shared_vals);
    if(my_lane == 0) warp_sum_arr[my_warp] = my_result;
    __syncthreads();

    if(my_warp == 0)
    {
        if(threadIdx.x >= blockDim.x / WARPSZ)
            warp_sum_arr[threadIdx.x] = 0.0;
        blk_result = Shared_mem_sum(warp_sum_arr);
    }

    if(threadIdx.x == 0) atomicAdd(lag, blk_result);
}

void Get_args(
    const int argc,
    char* argv[],
    int* slice_num,
    int* input_flag,
    float* x_input
)
{
    if(argc != 3)
    {
        *input_flag = 0;
        printf("An error message\n");
    }

    *slice_num = strtol(argv[1], NULL, 10);
    *x_input = strtol(argv[2], NULL, 10);
    
    if(*slice_num > MAX_BLKSZ)
    {
        *input_flag = 0;
        printf("An error message\n");
    }
}

int main(int argc,char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << std::endl;

    int slice_num;
    float h;
    float a = 0;
    float b = acos(-1.0);
    int input_flag = 1;
    float* lag_res;
    float x_input;
    float* d_lag_based_arr_p;
    
    Get_args(argc, argv, &slice_num, &input_flag, &x_input);
    if(input_flag == 0)
    {
        return 1;
    }

    h = (b - a) / (slice_num - 1);
    // printf("a: %f, b:%f h: %f \n", a, b, h);

    cudaMallocManaged(&d_lag_based_arr_p, slice_num* sizeof(float));
    cudaMallocManaged(&lag_res, sizeof(float));

    lagrangeGPU<<<BLOCK_COUNT, MAX_BLKSZ>>>(a, b, h, x_input, slice_num, lag_res, d_lag_based_arr_p);

    cudaError_t err = cudaDeviceSynchronize();
    
    for(int k = 0; k < slice_num; k++)
    {
        printf("L[%d] = %f \n", k, d_lag_based_arr_p[k]);
    }

    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}