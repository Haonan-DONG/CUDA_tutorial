#include <cuda_runtime.h>
#include <stdio.h>
#include "global.h"
#include <iostream>

void sumArrays(float *a, float *b, float *res, const int size)
{
    for (int i = 0; i < size; i++)
    {
        res[i] = a[i] + b[i];
    }
}

__global__ void sumArraysGPU(float *a, float *b, float *res, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
    {
        res[tid] = a[tid] + b[tid];

        // here the blockdim and griddim took the whole size apart equally.
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32 * 32 * 32 * 32 * 32 * 4;
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(float) * nElem;
    float *a_h = (float *)malloc(nByte);
    float *b_h = (float *)malloc(nByte);
    float *res_h = (float *)malloc(nByte);
    float *res_from_gpu_h = (float *)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float **)&a_d, nByte));
    CHECK(cudaMalloc((float **)&b_d, nByte));
    CHECK(cudaMalloc((float **)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    // remaining
    dim3 block(1024);
    dim3 grid(64);
    double iStart, iElaps;
    iStart = cpuSecond();

    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("Execution configuration<<<%d,%d>>>\n", block.x, grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    clock_t start_cpu, stop_cpu;
    float time_cpu;
    start_cpu = clock();
    sumArrays(a_h, b_h, res_h, nElem);
    stop_cpu = clock();

    time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

    std::cout << "cpu time : " << time_cpu << "\t"
              << "gpu time : " << iElaps << std::endl;

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}