# CUDA Tutorial
This is a basic tutorial for learning CUDA. The motivation is to present several examples for indicating the basic usage by CUDA, since there exists many blogs for the principle while the examples is deficient.

## How to use
```shell
# using the nvcc compiler
nvcc -g -G *.cu -o main

# run the code.
./main
```

## Note
All the code is tested on a RTX 3090 Card (SM_86 with CUDA 11.1).

## Source
### Demo 1 : sum_1d_large_arrays.cu
1. Introduce

This demo introduces how to add two large arrays (larger than the maximum thread number, eg, 1024).

2. Notice
- Using the while loop to operate the parallel function to add the large arrays.
- Using **CUDASynchronize()** to calculate the time on GPU, separately.

3. Remaining
- [ ] Why the maximum thread for RTX 3090 here is 1024 in one dim.

### Demo 2 : sum_2d_large_arrays.cu
1. Introduce

This demo introduces how to add two large 2d matrix.

## To-do List
- [x] Fixed the bug for the memory.
- [x] Create the blog for each demo.
- [ ] Image Convolution Process.
- [ ] PatchMatchStereo Algorithm by CUDA.


## Acknowledge
- [Sheng Tan's repo](https://github.com/Tony-Tan/CUDA_Freshman)
- [Caltech 2022](http://courses.cms.caltech.edu/cs179/)