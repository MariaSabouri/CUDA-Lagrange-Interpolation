#include <iostream>
#include <vector>
#include <cmath>

/*
g++ serial_lagrange.cpp -o serial_lagrange
./serial_lagrange
*/


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
    // return sin(x);
    return x;
}

void lagrange(
    const float a,
    const int n,
    const int query_n,
    const float h,
    const float query_h,
    float* result
)
{
    for(int q = 0; q < query_n; ++q)
    {    
        result[q] = 0.0f;
        float x_value = a + q * query_h; 
        for (int i = 0; i < n; ++i)
        {
            double basis = 1.0;
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
    }
}

int main(int argc,char* argv[])
{
    int slice_num, query_num;
    float h, query_h;
    float a = 0;
    float b = acos(-1.0);
    int input_flag = 1;

    Get_args(argc, argv, slice_num, &input_flag, query_num);
    if(input_flag == 0)
    {
        return 1;
    }

    query_h = (b - a) / (query_num - 1);
    h = (b - a) / (slice_num - 1);
    
    float* lag_res = new float[query_num];
    lagrange(a, slice_num, query_num, h, query_h, lag_res);
    for (int i = 0; i < query_num; ++i)
    {
        printf("lag[%d] = %f\n",i,lag_res[i]);
    }
    
/* 
    std::vector<float> lag_res(query_num);
    float* lag_res = (float*)malloc(query_num * sizeof(float));
    
    free(lag_res);
    */
    
    delete[] lag_res;
    return 0;
}