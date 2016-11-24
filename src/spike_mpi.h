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

#ifndef SPIKEMPIH
#define SPIKEMPIH

extern __device__ int NExternalTargetHost;
extern __device__ int MaxSpikePerHost;

extern int *d_ExternalSpikeNum;
extern __device__ int *ExternalSpikeNum;

extern int *d_ExternalSpikeSourceNeuron; // [MaxSpikeNum];
extern __device__ int *ExternalSpikeSourceNeuron;

extern float *d_ExternalSpikeHeight; // [MaxSpikeNum];
extern __device__ float *ExternalSpikeHeight;

extern int *d_ExternalTargetSpikeNum;
extern __device__ int *ExternalTargetSpikeNum;

extern int *d_ExternalTargetSpikeNeuronId;
extern __device__ int *ExternalTargetSpikeNeuronId;

extern float *d_ExternalTargetSpikeHeight;
extern __device__ float *ExternalTargetSpikeHeight;

extern int *d_NExternalNeuronTargetHost;
extern __device__ int *NExternalNeuronTargetHost;

extern int **d_ExternalNeuronTargetHostId;
extern __device__ int **ExternalNeuronTargetHostId;

extern int **d_ExternalNeuronId;
extern __device__ int **ExternalNeuronId;

//extern int *d_ExternalSourceSpikeNum;
//extern __device__ int *ExternalSourceSpikeNum;

extern int *d_ExternalSourceSpikeNeuronId;
extern __device__ int *ExternalSourceSpikeNeuronId;

extern float *d_ExternalSourceSpikeHeight;
extern __device__ float *ExternalSourceSpikeHeight;

__device__ void PushExternalSpike(int i_source, float height);

__global__ void SendExternalSpike();

__global__ void ExternalSpikeReset();

__global__ void DeviceExternalSpikeInit(int n_hosts,
					int max_spike_per_host,
		      			int *ext_spike_num,
					int *ext_spike_source_neuron,
                                        float *ext_spike_height,
					int *ext_target_spike_num,
					int *ext_target_spike_neuron_id,
                                        float *ext_target_spike_height,
					int *n_ext_neuron_target_host,
					int **ext_neuron_target_host_id,
					int **ext_neuron_id
					);

__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id,
                                    float *spike_height);

#endif
