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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "spike_generator.h"
#include "spike_buffer.h"
#include "cuda_error.h"

using namespace std;

__device__ int *SpikeGeneratorSpikeNum;

__device__ int *SpikeGeneratorSpikeIdx;

__device__ int **SpikeGeneratorTimeIdx;

__device__ float **SpikeGeneratorHeight;

__global__
void SpikeGeneratorUpdate(int i_node_0, int n_nodes, int i_time)
{
  int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_node < n_nodes) {
    if (SpikeGeneratorSpikeNum[i_node] > 0) { 
      int i_spike = SpikeGeneratorSpikeIdx[i_node];
      if (i_spike<SpikeGeneratorSpikeNum[i_node]
          && SpikeGeneratorTimeIdx[i_node][i_spike]==i_time) {
	int i_node_abs = i_node_0 + i_node;
	float height = SpikeGeneratorHeight[i_node][i_spike];
	PushSpike(i_node_abs, height);
	SpikeGeneratorSpikeIdx[i_node]++;
      }
    }
  }
}

__global__
void DeviceSpikeGeneratorInit(int *d_n_spikes, int *d_i_spike,
			      int **d_spike_time_idx,
			      float **d_spike_height)
{
  SpikeGeneratorSpikeNum = d_n_spikes;
  SpikeGeneratorSpikeIdx = d_i_spike;
  SpikeGeneratorTimeIdx = d_spike_time_idx;
  SpikeGeneratorHeight = d_spike_height;
}

int SpikeGenerator::Init()
{
  h_spike_time_idx_ = new int*[n_nodes_];
  h_spike_height_ = new float*[n_nodes_];
  for (int i_node=0; i_node<n_nodes_; i_node++) {
    h_spike_time_idx_[i_node] = 0;
    h_spike_height_[i_node] = 0;
  }
  
  gpuErrchk(cudaMalloc(&d_n_spikes_, n_nodes_*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_i_spike_, n_nodes_*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_spike_time_idx_, n_nodes_*sizeof(int*)));
  gpuErrchk(cudaMalloc(&d_spike_height_, n_nodes_*sizeof(float*)));
  
  gpuErrchk(cudaMemset(d_n_spikes_, 0, n_nodes_*sizeof(int)));
  gpuErrchk(cudaMemset(d_i_spike_, 0, n_nodes_*sizeof(int)));
  gpuErrchk(cudaMemset(d_spike_time_idx_, 0, n_nodes_*sizeof(int*)));
  gpuErrchk(cudaMemset(d_spike_height_, 0, n_nodes_*sizeof(float*)));
  
  DeviceSpikeGeneratorInit<<<1,1>>>(d_n_spikes_, d_i_spike_, d_spike_time_idx_,
			   d_spike_height_);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int SpikeGenerator::Free()
{
  for (int i_node=0; i_node<n_nodes_; i_node++) {
    if(h_spike_time_idx_[i_node] != 0) {
      gpuErrchk(cudaFree(h_spike_time_idx_[i_node]));
      gpuErrchk(cudaFree(h_spike_height_[i_node]));
    }
  }  
  gpuErrchk(cudaFree(d_n_spikes_));
  gpuErrchk(cudaFree(d_i_spike_));	    
  gpuErrchk(cudaFree(d_spike_time_idx_));
  gpuErrchk(cudaFree(d_spike_height_));

  delete[] h_spike_time_idx_;
  delete[] h_spike_height_;
  
  return 0;
}

SpikeGenerator::~SpikeGenerator()
{
  if (n_nodes_>0) {
    Free();
  }
}

SpikeGenerator::SpikeGenerator()
{
  n_nodes_ = 0;
}

int SpikeGenerator::Create(int i_node_0, int n_nodes, float time_min,
   float time_resolution)
{
  i_node_0_ = i_node_0;
  n_nodes_ = n_nodes;
  time_min_ = time_min;
  time_resolution_ = time_resolution;
  
  Init();
       
  return 0;
}

int SpikeGenerator::Update(int i_time)
{
   SpikeGeneratorUpdate<<<(n_nodes_+1023)/1024, 1024>>>(i_node_0_, n_nodes_,
                        i_time);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int SpikeGenerator::Set(int i_node, int n_spikes, float *spike_time,
      float *spike_height)
{
  if (n_spikes <=0) {
    throw ngpu_exception("Number of spikes must be greater than 0 "
			 "in spike generator setting");
  }
  
  cudaMemcpy(&d_n_spikes_[i_node], &n_spikes, sizeof(int),
	     cudaMemcpyHostToDevice);
  if (h_spike_time_idx_[i_node] != 0) {
    gpuErrchk(cudaFree(h_spike_time_idx_[i_node]));
    gpuErrchk(cudaFree(h_spike_height_[i_node]));
  }
  gpuErrchk(cudaMalloc(&h_spike_time_idx_[i_node], n_spikes*sizeof(int)));
  gpuErrchk(cudaMalloc(&h_spike_height_[i_node], n_spikes*sizeof(float)));

  cudaMemcpy(&d_spike_time_idx_[i_node], &h_spike_time_idx_[i_node],
	     sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(&d_spike_height_[i_node], &h_spike_height_[i_node], sizeof(float*),
	     cudaMemcpyHostToDevice);

  int *spike_time_idx = new int[n_spikes];
  for(int i=0; i<n_spikes; i++) {
    spike_time_idx[i] = (int)round((spike_time[i] - time_min_)
				   /time_resolution_);
    if (i>0 && spike_time_idx[i]<=spike_time_idx[i-1]) {
      throw ngpu_exception("Spike times must be ordered, and the difference "
			   "between\nconsecutive spikes must be >= the "
			   "time resolution");
    }
    //cout << "ti " << spike_time_idx[i] << endl;
    //cout << spike_time[i] << " " << time_min_ << endl;
      
  }
  
  cudaMemcpy(h_spike_time_idx_[i_node], spike_time_idx, n_spikes*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(h_spike_height_[i_node], spike_height, n_spikes*sizeof(float),
	     cudaMemcpyHostToDevice);

  return 0;
}
