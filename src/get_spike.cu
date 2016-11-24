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

#include "send_spike.h"
#include "spike_buffer.h"
#include "rk5.h"

extern __device__ int Aeif_i_node_0; 

__device__ double *GetSpikeArray;

double *d_GetSpikeArray;

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ int locate(uint val, uint *data, int n)
{
// remember to add a check that data[0]<=val<=data[n-1]
// what to do if not?
   int i_left = 0;
   int i_right = n-1;
   int i = (i_left+i_right)/2;
   while(i_right-i_left>1) {
      if (data[i] > val) i_right = i;
      else if (data[i]<val) i_left = i;
      else break;
      i=(i_left+i_right)/2;
   }

   return i;
}

__global__ void CollectSpikes(uint n_get_spikes, int nvar, int nparams)
{
  uint blockId   = blockIdx.y * gridDim.x + blockIdx.x;				
  uint array_idx = blockId * blockDim.x + threadIdx.x; 
  if (array_idx<n_get_spikes) {

    int i_spike = locate(array_idx, SpikeTargetNumSum, (*SpikeNum) + 1);
    int i_syn = array_idx - SpikeTargetNumSum[i_spike];
    int i_source = SpikeSourceIdx[i_spike];
    int i_conn = SpikeConnIdx[i_spike];
    float height = SpikeHeight[i_spike];
    int i_target = ConnectionGroupTargetNeuron[i_conn*NSpikeBuffer+i_source]
      [i_syn] - Aeif_i_node_0;
    unsigned char i_port = ConnectionGroupTargetPort[i_conn*NSpikeBuffer
						     +i_source][i_syn];
    float weight = ConnectionGroupTargetWeight[i_conn*NSpikeBuffer+i_source]
      [i_syn];
    
    // printf("handles spike %d src %d conn %d syn %d target %d"
    // " port %d weight %f\n",
    // i_spike, i_source, i_conn, i_syn, i_target,
    // i_port, weight);

    // IMPROVE THIS PART
    /////////////////////////////////////////////////////////////////
    int i = i_port*ARRAY_SIZE + i_target;
    int j = (N0_PARAMS + 3 + 4*i_port)*ARRAY_SIZE + i_target; // g0(i)
    double d_val = (double)(height*weight*ParamsArr[j]);
    atomicAddDouble(&GetSpikeArray[i], d_val); 
    ////////////////////////////////////////////////////////////////
  }
}

// improve using a grid
__global__ void GetSpikes(int n_ports)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < ARRAY_SIZE*n_ports) {
     int i_target = i_array % ARRAY_SIZE;
     int i_port = i_array / ARRAY_SIZE;
     int i = (N0_VAR + 1 + 2*i_port)*ARRAY_SIZE + i_target; // g1(i)
     double d_val = GetSpikeArray[i_array] + (double)YArr[i];  
     YArr[i] = (float)d_val;
  }
}
    // REMOVE THIS PART
    /////////////////////////////////////////////////////////////////
 //   int i = (N0_VAR + 1 + 2*i_port)*ARRAY_SIZE + i_target; // g1(i)
 //   int j = (N0_PARAMS + 3 + 4*i_port)*ARRAY_SIZE + i_target; // g0(i)
 //   atomicAdd(&YArr[i], height*weight*ParamsArr[j]); 
    ////////////////////////////////////////////////////////////////

__global__
void DeviceInitGetSpikeArray(double *get_spike_array)
{
  GetSpikeArray = get_spike_array;
}

int InitGetSpikeArray(int n_neurons, int n_ports)
{
  gpuErrchk(cudaMalloc(&d_GetSpikeArray, n_neurons*n_ports*sizeof(double)));
  DeviceInitGetSpikeArray<<<1, 1>>>(d_GetSpikeArray);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int ClearGetSpikeArray(int n_neurons, int n_ports)
{
  gpuErrchk(cudaMemset(d_GetSpikeArray, 0, n_neurons*n_ports*sizeof(double)));

  return 0;
}

int FreeGetSpikeArray()
{
  gpuErrchk(cudaFree(d_GetSpikeArray));

  return 0;
}

