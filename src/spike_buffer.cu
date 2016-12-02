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

#include "cuda_error.h"
#include "spike_buffer.h"
#include "connect.h"
#include "send_spike.h"
#include "spike_mpi.h"

using namespace std;

__device__ int MaxSpikeBufferSize;
__device__ int NSpikeBuffer;
__device__ int MaxDelayNum;

int *d_ConnectionGroupSize; // [NSpikeBuffer];
__device__ int *ConnectionGroupSize; // [NSpikeBuffer];
// ConnectionGroupSize[i_spike_buffer]
// where i_spike_buffer is the source neuron index
// Output connections from the source neuron are organized in groups
// All connection of a group have the same delay

int *d_ConnectionGroupDelay; // [NSpikeBuffer*MaxDelayNum];
__device__ int *ConnectionGroupDelay; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupDelay[i_delay*NSpikeBuffer+i_spike_buffer];
// delay associated to all connections of this group

int *d_ConnectionGroupTargetSize; // [NSpikeBuffer*MaxDelayNum];
__device__ int *ConnectionGroupTargetSize; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetSize[i_delay*NSpikeBuffer+i_spike_buffer];
// number of output connections in the group i_delay

int **d_ConnectionGroupTargetNeuron; // [NSpikeBuffer*MaxDelayNum];
__device__ int **ConnectionGroupTargetNeuron; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetNeuron[i_delay*NSpikeBuffer+i_spike_buffer];
// is a pointer to an integer array of size ConnectionGroupTargetSize
// that contains the indexes of the target neurons

unsigned char **d_ConnectionGroupTargetPort; // [NSpikeBuffer*MaxDelayNum];
__device__ unsigned char **ConnectionGroupTargetPort;
// [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetPort[i_delay*NSpikeBuffer+i_spike_buffer];
// Connection target port

float **d_ConnectionGroupTargetWeight; // [NSpikeBuffer*MaxDelayNum];
__device__ float **ConnectionGroupTargetWeight; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetWeight[i_delay*NSpikeBuffer+i_spike_buffer];
// Connection weight

//////////////////////////////////////////////////////////////////////

int *d_SpikeBufferSize; // [NSpikeBuffer];
__device__ int *SpikeBufferSize; // [NSpikeBuffer];
// SpikeBufferSize[i_spike_buffer];
// where i_spike_buffer is the source neuron index
// number of spikes stored in the buffer

int *d_SpikeBufferTimeIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ int *SpikeBufferTimeIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferTimeIdx[i_spike*NSpikeBuffer+i_spike_buffer];
// time index of the spike

int *d_SpikeBufferConnIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ int *SpikeBufferConnIdx; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferConnIdx[i_spike*NSpikeBuffer+i_spike_buffer];
// index of the next connection group that will emit this spike

float *d_SpikeBufferHeight; // [NSpikeBuffer*MaxSpikeBufferNum];
__device__ float *SpikeBufferHeight; // [NSpikeBuffer*MaxSpikeBufferNum];
// SpikeBufferHeight[i_spike*NSpikeBuffer+i_spike_buffer];
// spike height


__device__ void PushSpike(int i_spike_buffer, float height)
{
  PushExternalSpike(i_spike_buffer, height);
  if (ConnectionGroupSize[i_spike_buffer]>0) {
    int Ns = SpikeBufferSize[i_spike_buffer]; 
    if (Ns>=MaxSpikeBufferSize) {
      printf("Maximum number of spikes in spike buffer exceeded"
	     " for spike buffer %d\n", i_spike_buffer);
      //exit(0);
    }
    SpikeBufferSize[i_spike_buffer]++;
    for (int is=Ns; is>0; is--) {
      SpikeBufferTimeIdx[is*NSpikeBuffer+i_spike_buffer] =
	SpikeBufferTimeIdx[(is-1)*NSpikeBuffer+i_spike_buffer];
      SpikeBufferConnIdx[is*NSpikeBuffer+i_spike_buffer] =
	SpikeBufferConnIdx[(is-1)*NSpikeBuffer+i_spike_buffer];
      SpikeBufferHeight[is*NSpikeBuffer+i_spike_buffer] =
	SpikeBufferHeight[(is-1)*NSpikeBuffer+i_spike_buffer];
    }
    SpikeBufferTimeIdx[i_spike_buffer] = 0;
    SpikeBufferConnIdx[i_spike_buffer] = 0;
    SpikeBufferHeight[i_spike_buffer] = height;
  }
}

__global__ void SpikeBufferUpdate()
{
  int i_spike_buffer = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("NS %d\n", NSpikeBuffer);
  if (i_spike_buffer<NSpikeBuffer) {
    int Ns = SpikeBufferSize[i_spike_buffer];
    //printf("is %d  Ns %d\n", i_spike_buffer, Ns);
    for (int is=0; is<Ns; is++) {
      int i_conn = SpikeBufferConnIdx[is*NSpikeBuffer+i_spike_buffer];
      if (SpikeBufferTimeIdx[is*NSpikeBuffer+i_spike_buffer] == 
	  ConnectionGroupDelay[i_conn*NSpikeBuffer+i_spike_buffer]) {
	// spike time matches connection group delay
	float height = SpikeBufferHeight[is*NSpikeBuffer+i_spike_buffer];
	SendSpike(i_spike_buffer, i_conn, height,
		  ConnectionGroupTargetSize[i_conn*NSpikeBuffer
					    +i_spike_buffer]);

	if (is==Ns-1 && i_conn>=ConnectionGroupSize[i_spike_buffer]-1) {
	  // we don't need any more to keep track of the last spike
	  SpikeBufferSize[i_spike_buffer]--; // so remove it from buffer
	}
	else {
	  // increase index of the next conn. group that will emit this spike
	  SpikeBufferConnIdx[is*NSpikeBuffer+i_spike_buffer]++;
	}
      }
      SpikeBufferTimeIdx[is*NSpikeBuffer+i_spike_buffer]++;
      // increase time index
    }
  }
}

int SpikeBufferInit(NetConnection *net_connection, int max_spike_buffer_size)
{
  int n_spike_buffers = net_connection->connection_.size();
  int max_delay_num = net_connection->MaxDelayNum();
  
  int *h_ConnectionGroupSize = new int[n_spike_buffers];
  int *h_ConnectionGroupDelay = new int[n_spike_buffers*max_delay_num];
  int *h_ConnectionGroupTargetSize = new int[n_spike_buffers*max_delay_num];
  int **h_ConnectionGroupTargetNeuron = new int*[n_spike_buffers*max_delay_num];
  unsigned char **h_ConnectionGroupTargetPort =
    new unsigned char*[n_spike_buffers*max_delay_num];
  float **h_ConnectionGroupTargetWeight = new float*[n_spike_buffers
						     *max_delay_num];

  gpuErrchk(cudaMalloc(&d_ConnectionGroupSize, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupDelay,
		       n_spike_buffers*max_delay_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetSize,
		       n_spike_buffers*max_delay_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferSize, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferTimeIdx,
		       n_spike_buffers*max_spike_buffer_size*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferConnIdx,
		       n_spike_buffers*max_spike_buffer_size*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeBufferHeight,
		       n_spike_buffers*max_spike_buffer_size*sizeof(float)));
  gpuErrchk(cudaMemset(d_SpikeBufferSize, 0, n_spike_buffers*sizeof(int)));

  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetNeuron,
		     n_spike_buffers*max_delay_num*sizeof(int*)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetPort,
		     n_spike_buffers*max_delay_num*sizeof(unsigned char*)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetWeight,
		     n_spike_buffers*max_delay_num*sizeof(float*)));

  for (int i_source=0; i_source<n_spike_buffers; i_source++) {
    vector<ConnGroup> *conn = &(net_connection->connection_[i_source]);
    h_ConnectionGroupSize[i_source] = conn->size();
    for (unsigned int id=0; id<conn->size(); id++) {
      h_ConnectionGroupDelay[id*n_spike_buffers+i_source] = conn->at(id).delay;
     int n_target = conn->at(id).target_vect.size();
     h_ConnectionGroupTargetSize[id*n_spike_buffers+i_source] = n_target;

     gpuErrchk(cudaMalloc(&h_ConnectionGroupTargetNeuron
			  [id*n_spike_buffers+i_source],
			  n_target*sizeof(int)));
     gpuErrchk(cudaMalloc(&h_ConnectionGroupTargetPort
			  [id*n_spike_buffers+i_source],
			  n_target*sizeof(unsigned char)));
     gpuErrchk(cudaMalloc(&h_ConnectionGroupTargetWeight
			  [id*n_spike_buffers+i_source],
			  n_target*sizeof(float)));

     int *target_arr = new int[n_target];
     unsigned char *port_arr = new unsigned char[n_target];
     float *weight_arr = new float[n_target];
     for (int it=0; it<n_target; it++) {
       target_arr[it] = conn->at(id).target_vect[it].neuron;
       port_arr[it] = conn->at(id).target_vect[it].port;
       weight_arr[it] = conn->at(id).target_vect[it].weight;
     }
     cudaMemcpy(h_ConnectionGroupTargetNeuron[id*n_spike_buffers+i_source],
		target_arr, n_target*sizeof(int),
		cudaMemcpyHostToDevice);
     cudaMemcpy(h_ConnectionGroupTargetPort[id*n_spike_buffers+i_source],
		port_arr, n_target*sizeof(unsigned char),
		cudaMemcpyHostToDevice);
     cudaMemcpy(h_ConnectionGroupTargetWeight[id*n_spike_buffers+i_source],
		weight_arr, n_target*sizeof(float),
		cudaMemcpyHostToDevice);
     delete[] target_arr;
     delete[] port_arr;
     delete[] weight_arr;			       
   }
  }
  cudaMemcpy(d_ConnectionGroupSize, h_ConnectionGroupSize,
	     n_spike_buffers*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionGroupDelay, h_ConnectionGroupDelay,
	     n_spike_buffers*max_delay_num*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionGroupTargetSize, h_ConnectionGroupTargetSize,
	     n_spike_buffers*max_delay_num*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ConnectionGroupTargetNeuron, h_ConnectionGroupTargetNeuron,
	     n_spike_buffers*max_delay_num*sizeof(int*),
	     cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ConnectionGroupTargetPort, h_ConnectionGroupTargetPort,
	     n_spike_buffers*max_delay_num*sizeof(unsigned char*),
	     cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ConnectionGroupTargetWeight, h_ConnectionGroupTargetWeight,
	     n_spike_buffers*max_delay_num*sizeof(float*),
	     cudaMemcpyHostToDevice);

  DeviceSpikeBufferInit<<<1,1>>>(n_spike_buffers, max_delay_num,
			   max_spike_buffer_size,
			   d_ConnectionGroupSize, d_ConnectionGroupDelay,
			   d_ConnectionGroupTargetSize,
			   d_ConnectionGroupTargetNeuron,
			   d_ConnectionGroupTargetPort,
			   d_ConnectionGroupTargetWeight,
			   d_SpikeBufferSize, d_SpikeBufferTimeIdx,
			   d_SpikeBufferConnIdx, d_SpikeBufferHeight);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  delete[] h_ConnectionGroupSize;
  delete[] h_ConnectionGroupDelay;
  delete[] h_ConnectionGroupTargetNeuron;
  delete[] h_ConnectionGroupTargetPort;
  delete[] h_ConnectionGroupTargetWeight;

  return 0;
}

__global__ void DeviceSpikeBufferInit(int n_spike_buffers, int max_delay_num,
				int max_spike_buffer_size,
				int *conn_group_size, int *conn_group_delay,
				int *conn_group_target_size,
				int **conn_group_target_neuron,
				unsigned char **conn_group_target_port,
				float **conn_group_target_weight,
				int *spike_buffer_size, int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height)
{
  NSpikeBuffer = n_spike_buffers;
  MaxDelayNum = max_delay_num;
  MaxSpikeBufferSize = max_spike_buffer_size; 
  ConnectionGroupSize = conn_group_size;
  ConnectionGroupDelay = conn_group_delay;
  ConnectionGroupTargetSize = conn_group_target_size;
  ConnectionGroupTargetNeuron = conn_group_target_neuron;
  ConnectionGroupTargetPort = conn_group_target_port;
  ConnectionGroupTargetWeight = conn_group_target_weight;
  SpikeBufferSize = spike_buffer_size;
  SpikeBufferTimeIdx = spike_buffer_time;
  SpikeBufferConnIdx = spike_buffer_conn;
  SpikeBufferHeight = spike_buffer_height;
}

