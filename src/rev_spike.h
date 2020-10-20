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

#ifndef REVSPIKEH
#define REVSPIKEH

extern unsigned int *d_RevSpikeNum;
extern unsigned int *d_RevSpikeTarget;
extern int *d_RevSpikeNConn;

__global__ void RevSpikeReset();

__global__ void RevSpikeBufferUpdate(unsigned int n_node);

int RevSpikeInit(NetConnection *net_connection, int time_min_idx);

int RevSpikeFree();

int ResetConnectionSpikeTimeDown(NetConnection *net_connection);

int ResetConnectionSpikeTimeUp(NetConnection *net_connection);

#endif
