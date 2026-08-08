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
#define WARP_SZ 32
__device__ float Shared_mem_sum(float shared_vals[])
{
    int my_lane = threadIdx.x % warpSize;

    for(int diff = warpSize / 2; diff > 0; diff = diff / 2)
    {
        if (my_lane < diff)
        {
            shared_vals[my_lane] += shared_vals[my_lane + diff];
        }
        __syncwarp();
    }
    return shared_vals[0];
}

__device__ float U(float x)
{
    // return sin(x);
    return x;
}

__global__ void lagrangeGPU(
    const float a,
    const float h,
    const float query_h,
    const int n,
    float* lag
)
{
    __shared__ float thread_calcs[MAX_BLKSZ];
    __shared__ float warp_sum_arr[WARP_SZ];

    int query_i = blockIdx.x;
    int my_i = threadIdx.x;
    int my_warp = threadIdx.x / warpSize;
    int my_lane = threadIdx.x % warpSize;

    float* shared_vals = thread_calcs + my_warp * warpSize;
    float blk_result = 0.0;

    shared_vals[my_lane] = 0.0f;

    if (my_i < n)
    {
        float my_x = a + my_i * h;
        float li = 1.0f;

        for (int k = 0; k < n; k++)
        {
            if (k != my_i)
            {
                float x_k = a + k * h;
                float x_input = a + query_i * query_h;

                li *= (x_input - x_k) / (my_x - x_k);
            }
        }

        float my_y = U(my_x);
        shared_vals[my_lane] = my_y * li;
    }                                                
    
    float my_result = Shared_mem_sum(shared_vals);
    if(my_lane == 0) warp_sum_arr[my_warp] = my_result;
    __syncthreads();

    if(my_warp == 0)
    {
        if(threadIdx.x >= (blockDim.x + warpSize - 1) / warpSize)
            warp_sum_arr[threadIdx.x] = 0.0;
        blk_result = Shared_mem_sum(warp_sum_arr); 
    }

    if(threadIdx.x == 0){
      lag[query_i] = blk_result;
    } 
}

void Get_args(
    const int argc,
    char* argv[],
    int& slice_num,
    int* input_flag,
    int& query_num
)
{
    if(argc != 3)
    {
        *input_flag = 0;
        printf("An error message\n");
        return;
    }

    slice_num = strtol(argv[1], NULL, 10);
    query_num = strtol(argv[2], NULL, 10);
    
    if(slice_num < 2 || slice_num > MAX_BLKSZ)
    {
        *input_flag = 0;
        printf("slice_num must be between 2 and %d\n", MAX_BLKSZ);
        return;
    }

    if(query_num < 2)
    {
        *input_flag = 0;
        printf("query_num must be at least 2\n");
        return;
    }
}

int main(int argc,char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << std::endl;

    int slice_num, query_num;
    float h;
    float a = 0;
    float b = acos(-1.0);
    int input_flag = 1;
    float* lag_res;
    
    Get_args(argc, argv, slice_num, &input_flag, query_num);
    if(input_flag == 0)
    {
        return 1;
    }

    float query_h = (b - a) / (query_num - 1);

    h = (b - a) / (slice_num - 1);

    cudaMallocManaged(&lag_res, query_num * sizeof(float));
    cudaMemset(lag_res, 0, query_num * sizeof(float));
    /*
    query_num  → number of blocks 
    threads_per_block  → number of threads per block = blockDim.x
    */
    int threads_per_block = ((slice_num + 31) / 32) * 32;
    lagrangeGPU<<<query_num, threads_per_block>>>(a, h, query_h, slice_num, lag_res);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: "
                << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaDeviceSynchronize();

    for (int i = 0; i < query_num; i++)
    {
        printf("P(x[%d]) = %f\n", i, lag_res[i]);
    }

    if (err != cudaSuccess)
    {
        std::cerr << "Kernel execution error: "
                << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}