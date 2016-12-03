#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include "cuda_error.h"

unsigned int *curand_int(curandGenerator_t &gen, size_t n)
{
  unsigned int *devData, *hostData;
  // Allocate n integers on host
  hostData = (unsigned int *)calloc(n, sizeof(unsigned int));
  // Allocate n integers on device
  CUDA_CALL(cudaMalloc((void **)&devData, n * sizeof(unsigned int)));
  // Create pseudo-random number generator

  // Generate n integers on device
  CURAND_CALL(curandGenerate(gen, devData, n));
  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  // Cleanup
  CUDA_CALL(cudaFree(devData));
  
  return hostData;
}
