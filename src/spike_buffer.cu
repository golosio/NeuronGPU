/*
Copyright (C) 2020 Bruno Golosio
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

#include <config.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "spike_buffer.h"
#include "connect.h"
#include "send_spike.h"

#ifdef HAVE_MPI
#include "spike_mpi.h"
#endif

#define LAST_SPIKE_TIME_GUARD 0x70000000

extern __constant__ int NeuronGPUTimeIdx;

__device__ int MaxSpikeBufferSize;
__device__ int NSpikeBuffer;
__device__ int MaxDelayNum;

bool ConnectionSpikeTimeFlag;


float *d_LastSpikeHeight; // [NSpikeBuffer];
__device__ float *LastSpikeHeight; //

int *d_LastSpikeTimeIdx; // [NSpikeBuffer];
__device__ int *LastSpikeTimeIdx; //

float *d_ConnectionWeight; // [NConnection];
__device__ float *ConnectionWeight; //

unsigned char *d_ConnectionSynGroup; // [NConnection];
__device__ unsigned char *ConnectionSynGroup; //

unsigned short *d_ConnectionSpikeTime; // [NConnection];
__device__ unsigned short *ConnectionSpikeTime; //

int *d_ConnectionGroupSize; // [NSpikeBuffer];
__device__ int *ConnectionGroupSize; // [NSpikeBuffer];
// ConnectionGroupSize[i_spike_buffer]
// where i_spike_buffer is the source node index
// Output connections from the source node are organized in groups
// All connection of a group have the same delay

int *d_ConnectionGroupDelay; // [NSpikeBuffer*MaxDelayNum];
__device__ int *ConnectionGroupDelay; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupDelay[i_delay*NSpikeBuffer+i_spike_buffer];
// delay associated to all connections of this group

int *d_ConnectionGroupTargetSize; // [NSpikeBuffer*MaxDelayNum];
__device__ int *ConnectionGroupTargetSize; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetSize[i_delay*NSpikeBuffer+i_spike_buffer];
// number of output connections in the group i_delay

unsigned int **d_ConnectionGroupTargetNode; // [NSpikeBuffer*MaxDelayNum];
__device__ unsigned int **ConnectionGroupTargetNode;
// [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetNode[i_delay*NSpikeBuffer+i_spike_buffer];
// is a pointer to an integer array of size ConnectionGroupTargetSize
// that contains the indexes of the target nodes

unsigned char **d_ConnectionGroupTargetSynGroup; // [NSpikeBuffer*MaxDelayNum];
__device__ unsigned char **ConnectionGroupTargetSynGroup;
// [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetSynGroup[i_delay*NSpikeBuffer+i_spike_buffer];
// Connection target port


float **h_ConnectionGroupTargetWeight; //[NSpikeBuffer*MaxDelayNum];
float **d_ConnectionGroupTargetWeight; // [NSpikeBuffer*MaxDelayNum];
__device__ float **ConnectionGroupTargetWeight; // [NSpikeBuffer*MaxDelayNum];
// ConnectionGroupTargetWeight[i_delay*NSpikeBuffer+i_spike_buffer];
// Connection weight

unsigned short **d_ConnectionGroupTargetSpikeTime; //[NSpikeBuffer*MaxDelayNum];
__device__ unsigned short **ConnectionGroupTargetSpikeTime;
// ConnectionGroupTargetSpikeTime[i_delay*NSpikeBuffer+i_spike_buffer];
// Connection last spike time index

//////////////////////////////////////////////////////////////////////

int *d_SpikeBufferSize; // [NSpikeBuffer];
__device__ int *SpikeBufferSize; // [NSpikeBuffer];
// SpikeBufferSize[i_spike_buffer];
// where i_spike_buffer is the source node index
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


unsigned int *d_RevConnections; //[i] i=0,..., n_rev_conn - 1;
__device__ unsigned int *RevConnections;

int *d_TargetRevConnectionSize; //[i] i=0,..., n_neuron-1;
__device__ int *TargetRevConnectionSize;

unsigned int **d_TargetRevConnection; //[i][j] j=0,...,RevConnectionSize[i]-1
__device__ unsigned int **TargetRevConnection;


__device__ void PushSpike(int i_spike_buffer, float height)
{
  LastSpikeTimeIdx[i_spike_buffer] = NeuronGPUTimeIdx;
  LastSpikeHeight[i_spike_buffer] = height;

#ifdef HAVE_MPI
  if (NeuronGPUMpiFlag) {
    PushExternalSpike(i_spike_buffer, height);
  }
#endif
  
  if (ConnectionGroupSize[i_spike_buffer]>0) {
    int Ns = SpikeBufferSize[i_spike_buffer]; 
    if (Ns>=MaxSpikeBufferSize) {
      printf("Maximum number of spikes in spike buffer exceeded"
	     " for spike buffer %d\n", i_spike_buffer);
      //exit(0);
      return;
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

  if (i_spike_buffer<NSpikeBuffer) {
    int Ns = SpikeBufferSize[i_spike_buffer];
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

__global__ void InitLastSpikeTimeIdx(unsigned int n_spike_buffers,
				       int spike_time_idx)
{
  unsigned int i_spike_buffer = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_spike_buffer>=n_spike_buffers) {
    return;
  }
  LastSpikeTimeIdx[i_spike_buffer] = spike_time_idx;
}


int SpikeBufferInit(NetConnection *net_connection, int max_spike_buffer_size)
{
  unsigned int n_spike_buffers = net_connection->connection_.size();
  int max_delay_num = net_connection->MaxDelayNum();
  //printf("mdn: %d\n", max_delay_num);
  gpuErrchk(cudaMalloc(&d_LastSpikeTimeIdx, n_spike_buffers*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_LastSpikeHeight, n_spike_buffers*sizeof(float)));

  unsigned int n_conn = net_connection->StoredNConnections();
  unsigned int *h_conn_target = new unsigned int[n_conn];
  unsigned char *h_conn_syn_group = new unsigned char[n_conn];
  float *h_conn_weight = new float[n_conn];
  // unsigned short *h_conn_spike_time; //USELESS, REMOVE
  unsigned int *d_conn_target;
  gpuErrchk(cudaMalloc(&d_conn_target, n_conn*sizeof(unsigned int)));
  gpuErrchk(cudaMalloc(&d_ConnectionSynGroup, n_conn*sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&d_ConnectionWeight, n_conn*sizeof(float)));
  int *h_ConnectionGroupSize = new int[n_spike_buffers];
  int *h_ConnectionGroupDelay = new int[n_spike_buffers*max_delay_num];
  int *h_ConnectionGroupTargetSize = new int[n_spike_buffers*max_delay_num];
  unsigned int **h_ConnectionGroupTargetNode =
    new unsigned int*[n_spike_buffers*max_delay_num];
  unsigned char **h_ConnectionGroupTargetSynGroup =
    new unsigned char*[n_spike_buffers*max_delay_num];
  h_ConnectionGroupTargetWeight = new float*[n_spike_buffers
					     *max_delay_num];
  unsigned short **h_ConnectionGroupTargetSpikeTime = NULL;

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

  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetNode,
		     n_spike_buffers*max_delay_num*sizeof(unsigned int*)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetSynGroup,
		     n_spike_buffers*max_delay_num*sizeof(unsigned char*)));
  gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetWeight,
		     n_spike_buffers*max_delay_num*sizeof(float*)));

  if (ConnectionSpikeTimeFlag){
    //h_conn_spike_time = new unsigned short[n_conn];
    gpuErrchk(cudaMalloc(&d_ConnectionSpikeTime,
			 n_conn*sizeof(unsigned short)));
    //gpuErrchk(cudaMemset(d_ConnectionSpikeTime, 0,
    //			 n_conn*sizeof(unsigned short)));
    h_ConnectionGroupTargetSpikeTime
      = new unsigned short*[n_spike_buffers*max_delay_num];
    gpuErrchk(cudaMalloc(&d_ConnectionGroupTargetSpikeTime,
			 n_spike_buffers*max_delay_num
			 *sizeof(unsigned short*)));
  }
  
  unsigned int i_conn = 0;
  for (unsigned int i_source=0; i_source<n_spike_buffers; i_source++) {
    std::vector<ConnGroup> *conn = &(net_connection->connection_[i_source]);
    h_ConnectionGroupSize[i_source] = conn->size();
    for (unsigned int id=0; id<conn->size(); id++) {
      h_ConnectionGroupDelay[id*n_spike_buffers+i_source] = conn->at(id).delay;
     int n_target = conn->at(id).target_vect.size();
     h_ConnectionGroupTargetSize[id*n_spike_buffers+i_source] = n_target;

     h_ConnectionGroupTargetNode[id*n_spike_buffers+i_source]
       = &d_conn_target[i_conn];
     h_ConnectionGroupTargetSynGroup[id*n_spike_buffers+i_source]
       = &d_ConnectionSynGroup[i_conn];
     h_ConnectionGroupTargetWeight[id*n_spike_buffers+i_source]
       = &d_ConnectionWeight[i_conn];
     if (ConnectionSpikeTimeFlag){
       h_ConnectionGroupTargetSpikeTime[id*n_spike_buffers+i_source]
	 = &d_ConnectionSpikeTime[i_conn];
     }
     unsigned int *target_arr = &h_conn_target[i_conn];
     unsigned char *syn_group_arr = &h_conn_syn_group[i_conn];
     float *weight_arr = &h_conn_weight[i_conn];
     for (int it=0; it<n_target; it++) {
       unsigned int target = conn->at(id).target_vect[it].node;
       unsigned int port = conn->at(id).target_vect[it].port;
       target_arr[it] = (port << (24 + PORT_N_SHIFT)) | target;
       syn_group_arr[it] = conn->at(id).target_vect[it].syn_group;
       weight_arr[it] = conn->at(id).target_vect[it].weight;
     }
     i_conn += n_target;
   }
  }
  
  cudaMemcpy(d_conn_target, h_conn_target, n_conn*sizeof(unsigned int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionSynGroup, h_conn_syn_group,
	     n_conn*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionWeight, h_conn_weight, n_conn*sizeof(float),
	     cudaMemcpyHostToDevice);

  delete[] h_conn_weight;

  unsigned int n_rev_conn = 0;
  std::vector<std::vector<unsigned int> > rev_connections(n_spike_buffers);
  for (unsigned int i_conn=0; i_conn<n_conn; i_conn++) {
    unsigned char syn_group = h_conn_syn_group[i_conn];
    if (syn_group==1) { // TEMPORARY, TO BE IMPROVED
      n_rev_conn++;
      int target = h_conn_target[i_conn] & PORT_MASK;
      rev_connections[target].push_back(i_conn);
    }
  }
  
  delete[] h_conn_target;
  delete[] h_conn_syn_group;
  
  net_connection->SetNRevConnections(n_rev_conn);

  if (n_rev_conn>0) {
    unsigned int *h_rev_conn = new unsigned int[n_rev_conn];
    int *h_target_rev_conn_size = new int[n_spike_buffers];
    unsigned int **h_target_rev_conn = new unsigned int*[n_spike_buffers];
    
    gpuErrchk(cudaMalloc(&d_RevConnections, n_rev_conn*sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_TargetRevConnectionSize,
			 n_spike_buffers*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_TargetRevConnection, n_spike_buffers
			 *sizeof(unsigned int*)));

    unsigned int i_rev_conn = 0;
    for (unsigned int target=0; target<n_spike_buffers; target++) {
      h_target_rev_conn[target] = &d_RevConnections[i_rev_conn];
      int n_target_rev_conn = rev_connections[target].size();
      h_target_rev_conn_size[target] = n_target_rev_conn;
      for (int i=0; i<n_target_rev_conn; i++) {
	h_rev_conn[i_rev_conn] = rev_connections[target][i];
	i_rev_conn++;
      }
    }
    cudaMemcpy(d_RevConnections, h_rev_conn, n_rev_conn*sizeof(unsigned int),
	     cudaMemcpyHostToDevice);
    cudaMemcpy(d_TargetRevConnectionSize, h_target_rev_conn_size,
	       n_spike_buffers*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TargetRevConnection, h_target_rev_conn,
	       n_spike_buffers*sizeof(unsigned int*), cudaMemcpyHostToDevice);
    
    delete[] h_rev_conn;
    delete[] h_target_rev_conn_size;
    delete[] h_target_rev_conn;
  }
  
  cudaMemcpy(d_ConnectionGroupSize, h_ConnectionGroupSize,
	     n_spike_buffers*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionGroupDelay, h_ConnectionGroupDelay,
	     n_spike_buffers*max_delay_num*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ConnectionGroupTargetSize, h_ConnectionGroupTargetSize,
	     n_spike_buffers*max_delay_num*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ConnectionGroupTargetNode, h_ConnectionGroupTargetNode,
	     n_spike_buffers*max_delay_num*sizeof(unsigned int*),
	     cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ConnectionGroupTargetSynGroup, h_ConnectionGroupTargetSynGroup,
	     n_spike_buffers*max_delay_num*sizeof(unsigned char*),
	     cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ConnectionGroupTargetWeight, h_ConnectionGroupTargetWeight,
	     n_spike_buffers*max_delay_num*sizeof(float*),
	     cudaMemcpyHostToDevice);
  if(ConnectionSpikeTimeFlag) {
    cudaMemcpy(d_ConnectionGroupTargetSpikeTime,
	       h_ConnectionGroupTargetSpikeTime,
	       n_spike_buffers*max_delay_num*sizeof(unsigned short*),
	       cudaMemcpyHostToDevice);
  }
  
  DeviceSpikeBufferInit<<<1,1>>>(n_spike_buffers, max_delay_num,
			   max_spike_buffer_size,
			   d_LastSpikeTimeIdx, d_LastSpikeHeight,	 
			   d_ConnectionWeight, d_ConnectionSynGroup,
			   d_ConnectionSpikeTime,
			   d_ConnectionGroupSize, d_ConnectionGroupDelay,
			   d_ConnectionGroupTargetSize,
			   d_ConnectionGroupTargetNode,
			   d_ConnectionGroupTargetSynGroup,
			   d_ConnectionGroupTargetWeight,
			   d_ConnectionGroupTargetSpikeTime,
			   d_SpikeBufferSize, d_SpikeBufferTimeIdx,
			   d_SpikeBufferConnIdx, d_SpikeBufferHeight,
			   d_RevConnections, d_TargetRevConnectionSize,
			   d_TargetRevConnection
				 );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  InitLastSpikeTimeIdx
    <<<(n_spike_buffers+1023)/1024, 1024>>>
    (n_spike_buffers, LAST_SPIKE_TIME_GUARD);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaMemset(d_LastSpikeHeight, 0,
		       n_spike_buffers*sizeof(unsigned short)));

  delete[] h_ConnectionGroupSize;
  delete[] h_ConnectionGroupDelay;
  delete[] h_ConnectionGroupTargetNode;
  delete[] h_ConnectionGroupTargetSynGroup;
  //delete[] h_ConnectionGroupTargetWeight;
  if(h_ConnectionGroupTargetSpikeTime != NULL) {
    delete[] h_ConnectionGroupTargetSpikeTime;
  }
  
  return 0;
}

__global__ void DeviceSpikeBufferInit(int n_spike_buffers, int max_delay_num,
				int max_spike_buffer_size,
				int *last_spike_time_idx,
				float *last_spike_height,
				float *conn_weight,
				unsigned char *conn_syn_group,
				unsigned short *conn_spike_time,      
				int *conn_group_size, int *conn_group_delay,
				int *conn_group_target_size,
				unsigned int **conn_group_target_node,
				unsigned char **conn_group_target_syn_group,
				float **conn_group_target_weight,
				unsigned short **conn_group_target_spike_time,
				int *spike_buffer_size, int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height,
				unsigned int *rev_conn,
				int *target_rev_conn_size,
			        unsigned int **target_rev_conn
				      )
{
  NSpikeBuffer = n_spike_buffers;
  MaxDelayNum = max_delay_num;
  MaxSpikeBufferSize = max_spike_buffer_size;
  LastSpikeTimeIdx = last_spike_time_idx;
  LastSpikeHeight = last_spike_height;
  ConnectionWeight = conn_weight;
  ConnectionSynGroup = conn_syn_group;
  ConnectionSpikeTime = conn_spike_time;
  ConnectionGroupSize = conn_group_size;
  ConnectionGroupDelay = conn_group_delay;
  ConnectionGroupTargetSize = conn_group_target_size;
  ConnectionGroupTargetNode = conn_group_target_node;
  ConnectionGroupTargetSynGroup = conn_group_target_syn_group;
  ConnectionGroupTargetWeight = conn_group_target_weight;
  ConnectionGroupTargetSpikeTime = conn_group_target_spike_time;
  SpikeBufferSize = spike_buffer_size;
  SpikeBufferTimeIdx = spike_buffer_time;
  SpikeBufferConnIdx = spike_buffer_conn;
  SpikeBufferHeight = spike_buffer_height;
  RevConnections = rev_conn;
  TargetRevConnectionSize = target_rev_conn_size;
  TargetRevConnection = target_rev_conn;
}

