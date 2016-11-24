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

  MPI_Recv(int_val, 1, MPI_INT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_float(float *float_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(float_val, 1, MPI_FLOAT, sender_id, tag, MPI_COMM_WORLD, &Stat);

  return 0;
}

int ConnectMpi::MPI_Recv_uchar(unsigned char *uchar_val, int n, int sender_id)
{
  MPI_Status Stat;
  int tag = 1;

  MPI_Recv(uchar_val, 1, MPI_UNSIGNED_CHAR, sender_id, tag, MPI_COMM_WORLD,
	   &Stat);

  return 0;
}

int ConnectMpi::MPI_Send_int(int *int_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(int_val, 1, MPI_INT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_float(float *float_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(float_val, 1, MPI_FLOAT, target_id, tag, MPI_COMM_WORLD);

  return 0;
}

int ConnectMpi::MPI_Send_uchar(unsigned char *uchar_val, int n, int target_id)
{
  int tag = 1;

  MPI_Send(uchar_val, 1, MPI_UNSIGNED_CHAR, target_id, tag, MPI_COMM_WORLD);

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
  if (mpi_id_==0) return true;
  else return false;
}

int ConnectMpi::ReceiveCommands()
{
  int source_host_id;
  int source_neuron_id;
  int target_host_id;
  int target_neuron_id;
  unsigned char port_id;
  float weight;
  float delay;

  for(;;) {
    int command;
    MPI_Recv_int(&command, 1, mpi_master_);

    if (command == LOCAL_CONNECT) {
      MPI_Recv_int(&source_neuron_id, 1, mpi_master_);
      MPI_Recv_int(&target_neuron_id, 1, mpi_master_);
      MPI_Recv_uchar(&port_id, 1, mpi_master_);
      MPI_Recv_float(&weight, 1, mpi_master_);
      MPI_Recv_float(&delay, 1, mpi_master_);
      net_connection_->Connect(source_neuron_id, target_neuron_id, port_id,
			       weight, delay);

    }
    else if (command == SOURCE_CONNECT) {
      MPI_Recv_int(&source_neuron_id, 1, mpi_master_);
      MPI_Recv_int(&target_host_id, 1, mpi_master_);
      MPI_Recv_int(&target_neuron_id, 1, mpi_master_);
      MPI_Recv_uchar(&port_id, 1, mpi_master_);
      MPI_Recv_float(&weight, 1, mpi_master_);
      MPI_Recv_float(&delay, 1, mpi_master_);
      SourceConnect(source_neuron_id, target_host_id, target_neuron_id,
                    port_id, weight, delay);
    }
    else if (command == TARGET_CONNECT) {
      MPI_Recv_int(&source_host_id, 1, mpi_master_);
      MPI_Recv_int(&source_neuron_id, 1, mpi_master_);
      MPI_Recv_int(&target_neuron_id, 1, mpi_master_);
      MPI_Recv_uchar(&port_id, 1, mpi_master_);
      MPI_Recv_float(&weight, 1, mpi_master_);
      MPI_Recv_float(&delay, 1, mpi_master_);
      TargetConnect(source_host_id, source_neuron_id, target_neuron_id,
                    port_id, weight, delay);
    }
    else if (command == PRINT) {
      net_connection_->Print();
    }
    else if (command == QUIT) {
      break;
    }
  }

  return 0;
}

int ConnectMpi::SourceConnect(int source_neuron_id, int target_host_id,
                  int target_neuron_id, unsigned char port_id, float weight,
                  float delay)
{

  int remote_neuron_id = -1;
  for (vector<ExternalConnectionNode >::iterator it =
         extern_connection_[source_neuron_id].begin();
       it <  extern_connection_[source_neuron_id].end(); it++) {
    if ((*it).target_host_id == target_host_id) {
      remote_neuron_id = (*it).remote_neuron_id;
      break;
    }
  }
  MPI_Send_int(&remote_neuron_id, 1, target_host_id);
  if (remote_neuron_id == -1) {
    MPI_Recv_int(&remote_neuron_id, 1, target_host_id);
    ExternalConnectionNode conn_node = {target_host_id, remote_neuron_id};
    extern_connection_[source_neuron_id].push_back(conn_node);
  }

  return 0;
}

int ConnectMpi::TargetConnect(int source_host_id, int source_neuron_id,
                  int target_neuron_id, unsigned char port_id, float weight,
                  float delay)
{
  int remote_neuron_id;

  MPI_Recv_int(&remote_neuron_id, 1, source_host_id);
  if (remote_neuron_id == -1) {
    // Create remote connection node....
    remote_neuron_id = net_connection_->connection_.size();
    vector<ConnGroup> conn;
    net_connection_->connection_.push_back(conn);
    MPI_Send_int(&remote_neuron_id, 1, source_host_id);
  }
  net_connection_->Connect(remote_neuron_id, target_neuron_id,
			   port_id, weight, delay);

  return 0;
}

int ConnectMpi::RemoteConnect(int source_host_id, int source_neuron_id,
		  int target_host_id, int target_neuron_id,
		  unsigned char port_id, float weight, float delay)
{
  int command;
  
  if (source_host_id==mpi_master_ && target_host_id==mpi_master_) {
    net_connection_->Connect(source_neuron_id, target_neuron_id,
			     port_id, weight, delay);
    return 0;
  }
  else if (source_host_id == target_host_id) {
    command = LOCAL_CONNECT;
    MPI_Send_int(&command, 1, source_host_id);
    MPI_Send_int(&source_neuron_id, 1, source_host_id);
    MPI_Send_int(&target_neuron_id, 1, source_host_id);
    MPI_Send_uchar(&port_id, 1, source_host_id);
    MPI_Send_float(&weight, 1, source_host_id);
    MPI_Send_float(&delay, 1, source_host_id);
    return 0;
  }
  if (source_host_id!=mpi_master_) {
    command = SOURCE_CONNECT;
    MPI_Send_int(&command, 1, source_host_id);
    MPI_Send_int(&source_neuron_id, 1, source_host_id);
    MPI_Send_int(&target_host_id, 1, source_host_id);
    MPI_Send_int(&target_neuron_id, 1, source_host_id);
    MPI_Send_uchar(&port_id, 1, source_host_id);
    MPI_Send_float(&weight, 1, source_host_id);
    MPI_Send_float(&delay, 1, source_host_id);
  }
  if (target_host_id!=mpi_master_) {
    command = TARGET_CONNECT;
    MPI_Send_int(&command, 1, target_host_id);
    MPI_Send_int(&source_host_id, 1, target_host_id);
    MPI_Send_int(&source_neuron_id, 1, target_host_id);
    MPI_Send_int(&target_neuron_id, 1, target_host_id);
    MPI_Send_uchar(&port_id, 1, target_host_id);
    MPI_Send_float(&weight, 1, target_host_id);
    MPI_Send_float(&delay, 1, target_host_id);
  }
  if (source_host_id==mpi_master_) {
    SourceConnect(source_neuron_id, target_host_id, target_neuron_id, port_id,
		  weight, delay);
  }
  if (target_host_id==mpi_master_) {
    TargetConnect(source_host_id, source_neuron_id, target_neuron_id, port_id,
		  weight, delay);
  }

  return 0;
}

int ConnectMpi::RemoteConnectionPrint(int target_host_id)
{
  int command = PRINT;
  MPI_Send_int(&command, 1, target_host_id);

  return 0;
}

int ConnectMpi::Quit()
{
  int command = QUIT;
  for (int ith=0; ith<mpi_np_; ith++) {
    MPI_Send_int(&command, 1, ith);
  }

  return 0;
}
