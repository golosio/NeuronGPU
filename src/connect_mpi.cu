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

#include <iostream>
#include <cmath>
#include <stdlib.h>

#include "connect_mpi.h"

using namespace std;

int ConnectMpi::MPI_Recv_int(int *int_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(int_val, n, MPI_INT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_float(float *float_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(float_val, n, MPI_FLOAT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_uchar(unsigned char *uchar_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(uchar_val, n, MPI_UNSIGNED_CHAR, sender_id, tag, MPI_COMM_WORLD,
	   &Stat);

  return 0;
}

int ConnectMpi::MPI_Send_int(int *int_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(int_val, n, MPI_INT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_float(float *float_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(float_val, n, MPI_FLOAT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_uchar(unsigned char *uchar_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(uchar_val, n, MPI_UNSIGNED_CHAR, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MpiInit(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_np_);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id_);
  mpi_master_ = 0;
  
  return 0;
}

bool ConnectMpi::ProcMaster()
{
  if (mpi_id_==mpi_master_) return true;
  else return false;
}

int ConnectMpi::RemoteConnect(int i_source_host, int i_source_neuron,
			      int i_target_host, int i_target_neuron,
			      unsigned char i_port, float weight, float delay)
{
  int i_remote_neuron;
  
  if (mpi_id_==i_source_host && i_source_host==i_target_host) {
    return net_connection_->Connect(i_source_neuron, i_target_neuron, i_port, weight, delay);
  }
  else if (mpi_id_ == i_target_host) {
    MPI_Recv_int(&i_remote_neuron, 1, i_source_host);
    if (i_remote_neuron == -1) {
      // Create remote connection node....
      i_remote_neuron = net_connection_->connection_.size();
      vector<ConnGroup> conn;
      net_connection_->connection_.push_back(conn);
      MPI_Send_int(&i_remote_neuron, 1, i_source_host);
    }
    net_connection_->Connect(i_remote_neuron, i_target_neuron, i_port, weight, delay);
  }
  else if (mpi_id_ == i_source_host) {
    i_remote_neuron = -1;
    for (vector<ExternalConnectionNode >::iterator it =
	   extern_connection_[i_source_neuron].begin();
	 it <  extern_connection_[i_source_neuron].end(); it++) {
      if ((*it).target_host_id == i_target_host) {
	i_remote_neuron = (*it).remote_neuron_id;
	break;
      }
    }
    MPI_Send_int(&i_remote_neuron, 1, i_target_host);
    if (i_remote_neuron == -1) {
      MPI_Recv_int(&i_remote_neuron, 1, i_target_host);
      ExternalConnectionNode conn_node = {i_target_host, i_remote_neuron};
      extern_connection_[i_source_neuron].push_back(conn_node);
    }
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}
