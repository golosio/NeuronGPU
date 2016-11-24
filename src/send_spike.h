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

#ifndef SENDSPIKEH
#define SENDSPIKEH

extern int *d_SpikeNum;
extern int *d_SpikeSourceIdx;
extern int *d_SpikeConnIdx;
extern float *d_SpikeHeight;
extern uint *d_SpikeTargetNum;
extern uint *d_SpikeTargetNumSum;

//extern int *h_SpikeSourceIdx;
//extern int *h_SpikeConnIdx;
//extern float *h_SpikeHeight;
//extern int *h_SpikeTargetNum;
//extern int *h_SpikeTargetNumSum;

extern __device__ int MaxSpikeNum;
extern __device__ int *SpikeNum;
extern __device__ int *SpikeSourceIdx;
extern __device__ int *SpikeConnIdx;
extern __device__ float *SpikeHeight;
extern __device__ uint *SpikeTargetNum;
extern __device__ uint *SpikeTargetNumSum;

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				uint *spike_target_num,
				uint *spike_target_num_sum,
				int max_spike_num);

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num);

void SpikeInit(int max_spike_num);

__global__ void SpikeReset();

#endif
