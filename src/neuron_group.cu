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

#include <iostream>

#include "cuda_error.h"
#include "neuron_group.h"
#include "neural_gpu.h"

__constant__ NeuronGroup NeuronGroupArray[MAX_N_NEURON_GROUPS];
__device__ signed char *NeuronGroupMap;

__global__
void NeuronGroupMapInit(signed char *neuron_group_map)
{
  NeuronGroupMap = neuron_group_map;
}

int NeuralGPU::NeuronGroupArrayInit()
{
  gpuErrchk(cudaMalloc(&d_neuron_group_map_,
		       neuron_group_map_.size()*sizeof(signed char)));
  
  gpuErrchk(cudaMemcpyToSymbol(NeuronGroupArray, neuron_group_vect_.data(),
			       neuron_group_vect_.size()*sizeof(NeuronGroup)));

  gpuErrchk(cudaMemcpy(d_neuron_group_map_, neuron_group_map_.data(),
		       neuron_group_map_.size()*sizeof(signed char),
		       cudaMemcpyHostToDevice));
  NeuronGroupMapInit<<<1, 1>>>(d_neuron_group_map_);

  return 0;
}

int NeuralGPU::InsertNeuronGroup(int n_neurons, int n_receptors)
{
  double *d_get_spike_array = NULL;
  float *d_G0 = NULL;
  if (n_neurons*n_receptors > 0) {
    gpuErrchk(cudaMalloc(&d_get_spike_array, n_neurons*n_receptors
			 *sizeof(double)));
    gpuErrchk(cudaMalloc(&d_G0, n_neurons*n_receptors
			 *sizeof(float)));
  }
  NeuronGroup ng;
  ng.i_neuron_0_ = neuron_group_map_.size();
  ng.n_neurons_ = n_neurons;
  ng.n_receptors_ = n_receptors;
  ng.get_spike_array_ = d_get_spike_array;
  ng.G0_ = d_G0;
  
  int i_group = neuron_group_vect_.size();
  neuron_group_vect_.push_back(ng);
  neuron_group_map_.insert(neuron_group_map_.end(), n_neurons, i_group);
  
  return i_group;
}

int NeuralGPU::FreeNeuronGroupMap()
{
  gpuErrchk(cudaFree(d_neuron_group_map_));
	    
  return 0;
}
