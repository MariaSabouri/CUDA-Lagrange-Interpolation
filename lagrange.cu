#include <iostream>
#include <cuda_runtime.h>
#define _USE_MATH_DEFINES
#include <math.h>
/*
nvcc lagrange.cu -o lagrange
./lagrange
*/
__global__ void lagrangeGPU()
{
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
    int* blk_ct,
    int* th_per_blk,
    int* input_flag

)
{
    if(argc != 5)
    {
        *input_flag = 0;
        printf("An error message\n");
    }

    *th_per_blk = strtol(argv[1], NULL, 10);
    *blk_ct = strtol(argv[2], NULL, 10);
    *slice_num = strtol(argv[3], NULL, 10);
    
    if(*slice_num > (*blk_ct)*(*th_per_blk))
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

    int th_per_blk, blk_ct, slice_num;
    float h;
    float a = 0;
    float b = acos(-1.0);
    int input_flag = 1;

    h = (b - a) / (slice_num - 1);
    
    Get_args(argc, argv, &slice_num, &blk_ct, &th_per_blk, &input_flag);
    if(input_flag == 0)
    {
        return 1;
    }

    lagrangeGPU<<<blk_ct, th_per_blk>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}