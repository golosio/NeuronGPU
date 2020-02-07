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

#include <stdio.h>

#include "neuralgpu.h"
#include "node_group.h"
#include "send_spike.h"
#include "spike_buffer.h"
				    //#include "rev_spike.h"
#include "cuda_error.h"

extern __constant__ int NeuralGPUTimeIdx;
extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ signed char *NodeGroupMap;

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
__device__ void NestedLoopFunction0(int i_spike, int i_syn)
{
  int i_source = SpikeSourceIdx[i_spike];
  int i_conn = SpikeConnIdx[i_spike];
  float height = SpikeHeight[i_spike];
  unsigned int target_port
    = ConnectionGroupTargetNode[i_conn*NSpikeBuffer + i_source][i_syn];
  int i_target = target_port & PORT_MASK;
  unsigned char port = (unsigned char)(target_port >> (PORT_N_SHIFT + 24));
  unsigned char syn_group
    = ConnectionGroupTargetSynGroup[i_conn*NSpikeBuffer + i_source][i_syn];
  float weight = ConnectionGroupTargetWeight[i_conn*NSpikeBuffer+i_source]
    [i_syn];
  //printf("handles spike %d src %d conn %d syn %d target %d"
  //" port %d weight %f\n",
  //i_spike, i_source, i_conn, i_syn, i_target,
  //port, weight);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NodeGroupMap[i_target];
  int i = port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);

  atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  if (syn_group>0) {
    ConnectionGroupTargetSpikeTime[i_conn*NSpikeBuffer+i_source][i_syn]
      = (unsigned short)(NeuralGPUTimeIdx & 0xffff);
    
    int Dt = NeuralGPUTimeIdx - LastSpikeTimeIdx[i_target];
    // THIS IS TEMPORARY, JUST FOR TESTING
    if (Dt>0 && Dt<100) {
      ConnectionGroupTargetWeight[i_conn*NSpikeBuffer+i_source][i_syn]
	= weight - Dt;
    }
  }
  ////////////////////////////////////////////////////////////////
}
///////////////

// improve using a grid
__global__ void GetSpikes(int i_group, int array_size, int n_port, int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step,
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_port) {
     int i_target = i_array % array_size;
     int port = i_array / array_size;
     int port_input = i_target*port_input_arr_step
       + port_input_port_step*port;
     int port_weight = i_target*port_weight_arr_step
       + port_weight_port_step*port;
     double d_val = (double)port_input_arr[port_input]
       + NodeGroupArray[i_group].get_spike_array_[i_array]
       * port_weight_arr[port_weight];

     port_input_arr[port_input] = (float)d_val;
  }
}

int NeuralGPU::ClearGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaMemset(bn->get_spike_array_, 0, bn->n_node_*bn->n_port_
			   *sizeof(double)));
    }
  }
  
  return 0;
}

int NeuralGPU::FreeGetSpikeArrays()
{
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    BaseNeuron *bn = node_vect_[i];
    if (bn->get_spike_array_ != NULL) {
      gpuErrchk(cudaFree(bn->get_spike_array_));
    }
  }
  
  return 0;
}
