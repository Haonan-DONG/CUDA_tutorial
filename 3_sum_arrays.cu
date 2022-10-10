#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
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
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32 * 32 * 32 * 32 * 32;
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

    dim3 block(1024);
    dim3 grid(256);
    float time_gpu;
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);

    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);

    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);

    cudaEventElapsedTime(&time_gpu, start_gpu, stop_gpu);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    printf("Execution configuration<<<%d,%d>>>\n", block.x, grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    clock_t start_cpu, stop_cpu;
    float time_cpu;
    start_cpu = clock();
    sumArrays(a_h, b_h, res_h, nElem);
    stop_cpu = clock();

    time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

    std::cout << "cpu time : " << time_cpu << "\t"
              << "gpu time : " << time_gpu << std::endl;

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