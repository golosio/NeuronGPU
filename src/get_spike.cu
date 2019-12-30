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
    [i_syn];
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
  int i = i_port*NeuronGroupArray[i_group].n_neurons_ + i_target
    - NeuronGroupArray[i_group].i_neuron_0_;
  double d_val = (double)(height*weight);

  atomicAddDouble(&NeuronGroupArray[i_group].get_spike_array_[i], d_val); 
  ////////////////////////////////////////////////////////////////
}
///////////////

// improve using a grid
__global__ void GetSpikes(int i_group, int array_size, int n_ports, int n_var,
			  float *receptor_weight_arr,
			  int receptor_weight_arr_step,
			  int receptor_weight_port_step, //float *y_arr)
			  float *receptor_input_arr,
			  int receptor_input_arr_step,
			  int receptor_input_port_step)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_ports) {
     int i_target = i_array % array_size;
     int i_port = i_array / array_size;
     //int i = i_target*n_var + N_SCAL_VAR + N_VECT_VAR*i_port + i_g1; // g1(i)
     int i_receptor_input = i_target*receptor_input_arr_step
       + receptor_input_port_step*i_port;
     int i_receptor_weight = i_target*receptor_weight_arr_step
       + receptor_weight_port_step*i_port;
     //if (i_array==0) {
     //  printf("npar, irw, rw %d %d %f\n",
     // N_SCAL_PARAMS + N_VECT_PARAMS*n_ports,
     //	      i_receptor_weight,
     //	      NeuronGroupArray[i_group].receptor_weight_arr_
     //	      [i_receptor_weight]);
     //     }
     double d_val = (double)receptor_input_arr[i_receptor_input] // (double)y_arr[i]
       + NeuronGroupArray[i_group].get_spike_array_[i_array]
       * receptor_weight_arr[i_receptor_weight];

     //y_arr[i] =
     receptor_input_arr[i_receptor_input] = (float)d_val;
  }
}

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
