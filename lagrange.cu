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

__global__ void lagrangeGPU(
    const float a,
    const float b,
    const float h,
    const int n,
    float* lag
)
{
    __shared__ float thread_calcs[MAX_BLKSZ];
    __shared__ float warp_multi_arr[WARPSZ];

    int my_i = blockDim.x * blockIdx.x + threadIdx.x;
    int my_warp = threadIdx.x / WARPSZ;
    int my_lane = threadIdx.x % WARPSZ;
    float* shared_vals = thread_calcs + my_warp * WARPSZ;
    float blk_result = 0.0;

    shared_vals[my_lane] = 0.0f;
    if(0 < my_i && my_i < n)
    {
        //TODO
    }

    printf("Hello from GPU! Block %d Thread %d\n",
           blockIdx.x, threadIdx.x);
}

__device__ float U(float x)
{
    return sin(x);
}

void Get_args(
    const int argc,
    char* argv[],
    int* slice_num,
    int* input_flag
)
{
    if(argc != 5)
    {
        *input_flag = 0;
        printf("An error message\n");
    }

    *slice_num = strtol(argv[1], NULL, 10);
    
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


    h = (b - a) / (slice_num - 1);
    
    Get_args(argc, argv, &slice_num, &input_flag);
    if(input_flag == 0)
    {
        return 1;
    }

    cudaMallocManaged(&lag_res, sizeof(float));

    lagrangeGPU<<<BLOCK_COUNT, MAX_BLKSZ>>>(a, b, h, slice_num, lag_res);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}