#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_world()
{
    printf("cmake cuda test sucess!\n");
}

void kernel_hello_world()
{
    hello_world<<<2, 5>>>();
    cudaDeviceReset();
}