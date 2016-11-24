/*
Copyright (C) 2016 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include "poisson.h"
#include "spike_buffer.h"
#include "cuda_error.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
      return EXIT_FAILURE;}} while(0)

__device__ unsigned int *PoissonData;

__global__ void PoissonUpdate(unsigned int *poisson_data)
{
  PoissonData = poisson_data;
}

__global__
void PoissonSendSpikes(int i_node_0, int n_nodes)
{
  int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node < n_nodes) {
    int i_node_abs = i_node_0 + i_node;
    unsigned int height = PoissonData[i_node];
    if (height>0) {
      PushSpike(i_node_abs, (float)height);
    }
  }
}

int PoissonGenerator::Init(unsigned int n)
{
  poisson_data_size_ = n;
  // Allocate n floats on device
  CUDA_CALL(cudaMalloc((void **)&dev_poisson_data_, n * sizeof(unsigned int)));
  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&random_generator_,
                                    CURAND_RNG_PSEUDO_DEFAULT));
  // Set seed
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(random_generator_, 1234ULL));

  return 0;
}

int PoissonGenerator::Generate()
{
  return Generate(n_steps_);
}

int PoissonGenerator::Generate(int max_n_steps)
{
  if (max_n_steps <= 0) {
    more_steps_ = n_steps_;
  }
  else {
    more_steps_ = min(n_steps_, max_n_steps);
  }
  // Generate N floats on device
  CURAND_CALL(curandGeneratePoisson(random_generator_, dev_poisson_data_,
				    n_nodes_*more_steps_, lambda_));

  return 0;
}

int PoissonGenerator::Free()
{
  CURAND_CALL(curandDestroyGenerator(random_generator_));
  CUDA_CALL(cudaFree(dev_poisson_data_));

  return 0;
}

PoissonGenerator::~PoissonGenerator()
{
  //Free();
}

PoissonGenerator::PoissonGenerator()
{
  buffer_size_ = 100000;
  n_nodes_ = 0;
}

int PoissonGenerator::Create(int i_node_0, int n_nodes, float lambda)
{
  i_node_0_ = i_node_0;
  n_nodes_ = n_nodes;
  lambda_ = lambda;
  
  n_steps_ = (buffer_size_ - 1)/n_nodes + 1;
  // with the above formula:
  // buffer_size <= n_nodes*n_steps <= buffer_size + n_nodes - 1
  Init(n_nodes_*n_steps_);
  i_step_ = 0;
       
  return 0;
}

int PoissonGenerator::Update(int max_n_steps)
{
  if (i_step_ == 0) {
    Generate(max_n_steps);
  }

  if (i_step_ == more_steps_) {
    fprintf(stderr, "Step index larger than maximum number of steps in poisson"
	    " generator\n");
    exit(0);
  }
  
  PoissonUpdate<<<1, 1>>>(&dev_poisson_data_[i_step_*n_nodes_]);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  PoissonSendSpikes<<<(n_nodes_+1023)/1024, 1024>>>(i_node_0_, n_nodes_);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  i_step_++;
  if (i_step_ == n_steps_) i_step_ = 0;

  return 0;
}

