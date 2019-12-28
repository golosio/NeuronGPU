/*
Copyright (C) 2019 Bruno Golosio
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

#include "neural_gpu.h"
#include "neuron_group.h"
#include "send_spike.h"
#include "spike_buffer.h"
#include "rk5.h"

extern __constant__ NeuronGroup NeuronGroupArray[];
extern __device__ signed char *NeuronGroupMap;

extern __device__ int Aeif_i_node_0; 
extern __device__ float *G0;

//__device__ double *GetSpikeArray;

//__device__ int N_NEURONS;

//double *d_GetSpikeArray;

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

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that collects the spikes
__device__ void NestedLoopFunction(int i_spike, int i_syn)
{
  int i_source = SpikeSourceIdx[i_spike];
  int i_conn = SpikeConnIdx[i_spike];
  float height = SpikeHeight[i_spike];
  int i_target = ConnectionGroupTargetNeuron[i_conn*NSpikeBuffer+i_source]
    [i_syn]; // - Aeif_i_node_0;
  //printf("i_target: %d\n", i_target);
  unsigned char i_port = ConnectionGroupTargetPort[i_conn*NSpikeBuffer
						   +i_source][i_syn];
  float weight = ConnectionGroupTargetWeight[i_conn*NSpikeBuffer+i_source]
    [i_syn];
  //printf("handles spike %d src %d conn %d syn %d target %d"
  //" port %d weight %f\n",
  //i_spike, i_source, i_conn, i_syn, i_target,
  //i_port, weight);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NeuronGroupMap[i_target];
  //printf("i_group: %d\n", i_group);
  int i = i_port*NeuronGroupArray[i_group].n_neurons_ + i_target
    - NeuronGroupArray[i_group].i_neuron_0_;
  double d_val = (double)(height*weight*G0[i]);

  //printf("in0: %d\n", Aeif_i_node_0);
  //printf("i_target, i_port, i_group, i: %d %d %d %d\n",
  //	 i_target, i_port, i_group, i);

  
  atomicAddDouble(&NeuronGroupArray[i_group].get_spike_array_[i], d_val); 
  ////////////////////////////////////////////////////////////////
}
///////////////

// improve using a grid
__global__ void GetSpikes(int i_group, int array_size, int n_ports, int n_var,
			  float *y_arr)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_ports) {
     int i_target = i_array % array_size;
     int i_port = i_array / array_size;
     int i = i_target*n_var + N0_VAR + 1 + 2*i_port; // g1(i)
     double d_val =
       NeuronGroupArray[i_group].get_spike_array_[i_array] + (double)y_arr[i];
     y_arr[i] = (float)d_val;
  }
}

//__global__
//void DeviceInitGetSpikeArray(double *get_spike_array, int n_neurons)
//{
//  GetSpikeArray = get_spike_array;
//  N_NEURONS = n_neurons;
//}

//int InitGetSpikeArray(int n_neurons, int n_ports)
//{
//  gpuErrchk(cudaMalloc(&d_GetSpikeArray, n_neurons*n_ports*sizeof(double)));
//  DeviceInitGetSpikeArray<<<1, 1>>>(d_GetSpikeArray, n_neurons);
//  gpuErrchk( cudaPeekAtLastError() );
//  gpuErrchk( cudaDeviceSynchronize() );

//  return 0;
//}

//int ClearGetSpikeArray(int n_neurons, int n_ports)
//{ 
//  gpuErrchk(cudaMemset(d_GetSpikeArray, 0, n_neurons*n_ports*sizeof(double)));
//
//  return 0;
//}

//int FreeGetSpikeArray()
//{
//  gpuErrchk(cudaFree(d_GetSpikeArray));

//  return 0;
//}

int NeuralGPU::ClearGetSpikeArrays()
{
  for (unsigned int i=0; i<neuron_group_vect_.size(); i++) {
    NeuronGroup ng = neuron_group_vect_[i];
    gpuErrchk(cudaMemset(ng.get_spike_array_, 0, ng.n_neurons_*ng.n_receptors_
			 *sizeof(double)));
  }
  
  return 0;
}

int NeuralGPU::FreeGetSpikeArrays()
{
  for (unsigned int i=0; i<neuron_group_vect_.size(); i++) {
    NeuronGroup ng = neuron_group_vect_[i];
    if (ng.n_neurons_*ng.n_receptors_ > 0) {
      gpuErrchk(cudaFree(ng.get_spike_array_));
    }
  }
  
  return 0;
}


