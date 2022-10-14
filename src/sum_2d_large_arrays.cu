#include <cuda_runtime.h>
#include <iostream>
#include "global.h"

void sum2DArrays(float **a, float **b, float **res, const int height, const int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
}

// void sum2DArraysGPU(float **a, float **b, float **res, const int height, const int width)
// {
//     int thread_x = threadIdx.x + threadIdx.x * blockDim.x;
//     int thread_y = threadIdx.y + threadIdx.y * blockIdx.y;

//     // be careful with the parallel.
//     res[thread_x][thread_y] = a[thread_x][thread_y] + b[thread_x][thread_y];
// }

int main()
{
    // the id of the device.
    int dev = 1;
    cudaSetDevice(dev);

    int nElem = 1024;
    std::cout << "The array size is " << nElem
              << "*" << nElem
              << ", total with " << nElem * nElem
              << std::endl;
    // allocate the size
    float **a = new float *[nElem];
    float **b = new float *[nElem];
    float **c = new float *[nElem];
    for (int i = 0; i < nElem; i++)
    {
        a[i] = new float[nElem];
        b[i] = new float[nElem];
        c[i] = new float[nElem]{0};
    }

    float(*a_gpu)[nElem], (*b_gpu)[nElem], (*c_gpu)[nElem];
    cudaMalloc((void **)&a_gpu, (nElem * nElem) * sizeof(int));
    cudaMalloc((void **)&b_gpu, (nElem * nElem) * sizeof(int));
    cudaMalloc((void **)&c_gpu, (nElem * nElem) * sizeof(int));

    initialData2D(a, nElem, nElem);
    initialData2D(b, nElem, nElem);

    // CHECK(cudaMemcpy(a, a_gpu, (nElem * nElem) * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(b, b_gpu, (nElem * nElem) * sizeof(int), cudaMemcpyHostToDevice));

    clock_t start_cpu, stop_cpu;
    float time_cpu;
    start_cpu = clock();
    sum2DArrays(a, b, c, nElem, nElem);
    stop_cpu = clock();

    time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

    std::cout << "CPU time : " << time_cpu << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}