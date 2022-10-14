cmake_minimum_required(VERSION 3.8)
project(CUDA_TUTORIAL)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)

include_directories(include)

set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86)

cuda_add_executable(1d_large_array src/sum_1d_large_arrays.cu)
cuda_add_executable(2d_large_array src/sum_2d_large_arrays.cu)