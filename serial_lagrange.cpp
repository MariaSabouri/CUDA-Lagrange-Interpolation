#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdio.h>

/*
g++ serial_lagrange.cpp -o serial_lagrange
./serial_lagrange
*/

void save_results(
    int slice_number,
    int block_size,
    int query_num,
    double elapsed,
    double mean_error,
    const char* version
)
{
    FILE *fp = fopen("result.csv", "a");

    if (fp == NULL)
    {
        perror("Error opening file");
        return;
    }

    fseek(fp, 0, SEEK_END);

    if (ftell(fp) == 0)
    {
        fprintf(fp,
            "slice_number,block_size,query_num,elapsed(us),mean_error,version\n");
    }

    fprintf(fp,
        "%d,%d,%d,%.10e,%.10e,%s\n",
        slice_number,
        block_size,
        query_num,
        elapsed,
        mean_error,
        version
    );

    fclose(fp);
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
    
    if(slice_num < 2)
    {
        *input_flag = 0;
        printf("slice_num must greater than %d\n", 1);
        return;
    }

    if(query_num < 2)
    {
        *input_flag = 0;
        printf("query_num must greater than %d\n", 1);
        return;
    }
}

float U(float x)
{
    return sin(x);
    // return x;
}

void lagrange(
    const float a,
    const int n,
    const int query_n,
    const float h,
    const float query_h,
    float* result,
    float& mean_lag_error
)
{
    for(int q = 0; q < query_n; ++q)
    {    
        result[q] = 0.0f;
        float x_value = a + q * query_h; 
        for (int i = 0; i < n; ++i)
        {
            float basis = 1.0f;
            float x_i = a + i * h;

            for (int j = 0; j < n; ++j)
            {
                if (i != j)
                {
                    float x_j = a + j * h;
                    basis *= (x_value - x_j) / (x_i - x_j);
                }
            }

            float y_i = U(x_i);
            result[q] += y_i * basis;
        }
        mean_lag_error += fabsf(result[q] - U(x_value));
    }
    mean_lag_error /= query_n;
}

int main(int argc,char* argv[])
{
    int slice_num, query_num;
    float h, query_h;
    float a = 0;
    float b = acos(-1.0);
    float mean_lag_error = 0.0f;
    int input_flag = 1;

    Get_args(argc, argv, slice_num, &input_flag, query_num);
    if(input_flag == 0)
    {
        return 1;
    }

    query_h = (b - a) / (query_num - 1);
    h = (b - a) / (slice_num - 1);
    
    float* lag_res = new float[query_num];

    auto start = std::chrono::high_resolution_clock::now();
    lagrange(a, slice_num, query_num, h, query_h, lag_res, mean_lag_error);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> elapsed = end - start;

    // for (int i = 0; i < query_num; ++i)
    // {
    //     printf("lag[%d] = %f\n",i,lag_res[i]);
    // }
    printf("Execution time: %f us\n", elapsed.count());
    printf("mean_error: %.12f \n", mean_lag_error);

    save_results(
    slice_num,
    0,
    query_num,
    elapsed.count(),
    mean_lag_error,
    "Serial");
    
/* 
    std::vector<float> lag_res(query_num);
    float* lag_res = (float*)malloc(query_num * sizeof(float));
    
    free(lag_res);
    */
    
    delete[] lag_res;
    return 0;
}