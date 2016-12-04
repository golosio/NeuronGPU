#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include "cuda_error.h"

unsigned int *curand_int(curandGenerator_t &gen, size_t n)
{
  unsigned int *dev_data;
  // Allocate n integers on host
  unsigned int *host_data = new unsigned int[n];
  
  // Allocate n integers on device
  CUDA_CALL(cudaMalloc((void **)&dev_data, n*sizeof(unsigned int)));
  // Create pseudo-random number generator

  // Generate n integers on device
  CURAND_CALL(curandGenerate(gen, dev_data, n));
  // Copy device memory to host
  CUDA_CALL(cudaMemcpy(host_data, dev_data, n*sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  // Cleanup
  CUDA_CALL(cudaFree(dev_data));
  
  return host_data;
}
