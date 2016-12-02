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

#ifndef SPIKEBUFFERH
#define SPIKEBUFFERH
#include "connect.h"

extern __device__ int MaxSpikeBufferSize;
extern __device__ int NSpikeBuffer;
extern __device__ int MaxDelayNum;

extern int *d_ConnectionGroupSize;
extern __device__ int *ConnectionGroupSize;
// Output connections from the source neuron are organized in groups
// All connection of a group have the same delay

extern int *d_ConnectionGroupDelay;
extern __device__ int *ConnectionGroupDelay;
// delay associated to all connections of this group

extern int *d_ConnectionGroupTargetSize;
extern __device__ int *ConnectionGroupTargetSize;
// number of output connections in the group i_delay

extern int **d_ConnectionGroupTargetNeuron;
extern __device__ int **ConnectionGroupTargetNeuron;
// is a pointer to an integer array of size ConnectionGroupTargetSize
// that contains the indexes of the target neurons

extern unsigned char **d_ConnectionGroupTargetPort;
extern __device__ unsigned char **ConnectionGroupTargetPort;
// Connection target port

extern float **d_ConnectionGroupTargetWeight;
extern __device__ float **ConnectionGroupTargetWeight;
// Connection weight

//////////////////////////////////////////////////////////////////////

extern int *d_SpikeBufferSize;
extern __device__ int *SpikeBufferSize;
// number of spikes stored in the buffer

extern int *d_SpikeBufferTimeIdx;
extern __device__ int *SpikeBufferTimeIdx;
// time index of the spike

extern int *d_SpikeBufferConnIdx;
extern __device__ int *SpikeBufferConnIdx;
// index of the next connection group that will emit this spike

extern float *d_SpikeBufferHeight;
extern __device__ float *SpikeBufferHeight;
// spike height

__device__ void PushSpike(int i_spike_buffer, float height);

__global__ void SpikeBufferUpdate();

__global__ void DeviceSpikeBufferInit(int n_spike_buffer, int max_delay_num,
				int max_spike_buffer_size,
				int *conn_group_size, int *conn_group_delay,
				int *conn_group_target_size,
				int **conn_group_target_neuron,
				unsigned char **conn_group_target_port,
				float **conn_group_target_weight,
				int *spike_buffer_size, int *spike_buffer_time,
				int *spike_buffer_conn,
				float *spike_buffer_height);
int SpikeBufferInit(NetConnection *net_connection, int max_spike_buffer_size);

#endif
