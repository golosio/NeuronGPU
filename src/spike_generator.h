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

#ifndef SPIKE_GENERATOR_H
#define SPIKE_GENERATOR_H

extern __device__ int *SpikeGeneratorSpikeNum;

extern __device__ int *SpikeGeneratorSpikeIdx;

extern __device__ int **SpikeGeneratorTimeIdx;

extern __device__ float **SpikeGeneratorHeight;

__global__
void SpikeGeneratorUpdate(int i_node_0, int n_nodes, int i_time);

__global__
void DeviceSpikeGeneratorInit(int *d_n_spikes, int *d_i_spike,
			      int **d_spike_time_idx,
			      float **d_spike_height);

class SpikeGenerator
{
 public:
  int i_node_0_;
  int n_nodes_;
  float time_resolution_;
  float time_min_;

  int *d_n_spikes_;
  int *d_i_spike_;	    
  int **d_spike_time_idx_;
  float **d_spike_height_;
  int **h_spike_time_idx_;
  float ** h_spike_height_;

  SpikeGenerator();

  ~SpikeGenerator();
    
    int Init();
    
    int Create(int i_node_0, int n_nodes, float min_time,
	       float time_resolution);
    
    int Free();
    
    int Update(int i_time);
    
    int Set(int i_node, int n_spikes, float *spike_time, float *spike_height);

};

#endif
