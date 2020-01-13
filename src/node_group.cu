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

#include <iostream>

#include "cuda_error.h"
#include "node_group.h"
#include "neuralgpu.h"

__constant__ NodeGroupStruct NodeGroupArray[MAX_N_NODE_GROUPS];
__device__ signed char *NodeGroupMap;

__global__
void NodeGroupMapInit(signed char *node_group_map)
{
  NodeGroupMap = node_group_map;
}

int NeuralGPU::NodeGroupArrayInit()
{
  gpuErrchk(cudaMalloc(&d_node_group_map_,
		       node_group_map_.size()*sizeof(signed char)));

  std::vector<NodeGroupStruct> ngs_vect;
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    NodeGroupStruct ngs;
    ngs.node_type_ = node_vect_[i]->node_type_;
    ngs.i_node_0_ = node_vect_[i]->i_node_0_;
    ngs.n_nodes_ = node_vect_[i]->n_nodes_;
    ngs.n_ports_ = node_vect_[i]->n_ports_;
    ngs.get_spike_array_ = node_vect_[i]->get_spike_array_;
    ngs_vect.push_back(ngs);
  }
  gpuErrchk(cudaMemcpyToSymbol(NodeGroupArray, ngs_vect.data(),
			       ngs_vect.size()*sizeof(NodeGroupStruct)));

  gpuErrchk(cudaMemcpy(d_node_group_map_, node_group_map_.data(),
		       node_group_map_.size()*sizeof(signed char),
		       cudaMemcpyHostToDevice));
  NodeGroupMapInit<<<1, 1>>>(d_node_group_map_);

  return 0;
}

double *NeuralGPU::InitGetSpikeArray (int n_nodes, int n_ports)
{
  double *d_get_spike_array = NULL;
  if (n_nodes*n_ports > 0) {
    gpuErrchk(cudaMalloc(&d_get_spike_array, n_nodes*n_ports
			 *sizeof(double)));
  }
  
  return d_get_spike_array;
}

int NeuralGPU::FreeNodeGroupMap()
{
  gpuErrchk(cudaFree(d_node_group_map_));
	    
  return 0;
}
