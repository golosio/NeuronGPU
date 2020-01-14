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

#ifndef CONNECTRULESH
#define CONNECTRULESH

#include <iostream>
#include "neuralgpu.h"
#include "connect_mpi.h"

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

template<>
int RemoteNode<int>::GetINode(int in)
{
  return i_node_ + in;
}

template<>
int RemoteNode<int*>::GetINode(int in)
{
  return *(i_node_ + in);
}

template <class T1, class T2>
int NeuralGPU::_Connect(T1 source, int n_source, T2 target, int n_target,
			ConnSpec &conn_spec, SynSpec &syn_spec)
{
  ////////////////////////
  // TO DO:
  // if (syn_spec.weight_distr_ != NULL) {
  //   syn_spec.weight_array_ = Distribytion(syn_spec.weight_distr, n);
  // }
  // if (syn_spec.delay_distr_ != NULL) {
  //   syn_spec.delay_array_ = Distribytion(syn_spec.delay_distr, n);
  // }
  
  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      std::cerr << "Number of source and target nodes must be equal "
	"for the one-to-one connection rule\n";
      exit(0);
    }
    return _ConnectOneToOne<T1, T2>(source, target, n_source, syn_spec);
    break;
  case ALL_TO_ALL:
    return _ConnectAllToAll<T1, T2>(source, n_source, target, n_target,
				    syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _ConnectFixedTotalNumber<T1, T2>(source, n_source, target, n_target,
				       conn_spec.total_num_, syn_spec);
    break;
  case FIXED_INDEGREE:
    return _ConnectFixedIndegree<T1, T2>(source, n_source, target, n_target,
				    conn_spec.indegree_, syn_spec);
    break;
  case FIXED_OUTDEGREE:
    //_ConnectFixedOutdegree<T1, T2>(source, n_source, target, n_target,
    //conn_spec.outdegree_, syn_spec);
    //break;
  default:
    std::cerr << "Unknown connection rule\n";
    exit(0);
  }
  return 0;
}

template<class T1, class T2>
int NeuralGPU::_SingleConnect(T1 source, int i_source, T2 target, int i_target,
			      int i_array, SynSpec &syn_spec)
{
  float weight;
  if (syn_spec.weight_array_ != NULL) {
    weight = syn_spec.weight_array_[i_array];
  }
  else {
    weight = syn_spec.weight_;
  }
  float delay;
  if (syn_spec.delay_array_ != NULL) {
    delay = syn_spec.delay_array_[i_array];
  }
  else {
    delay = syn_spec.delay_;
  }
  return _SingleConnect<T1, T2>(source, i_source, target, i_target,
				weight, delay, i_array, syn_spec);
}

template<class T1, class T2>
int NeuralGPU::_SingleConnect(T1 source, int i_source, T2 target, int i_target,
			      float weight, float delay, int i_array,
			      SynSpec &syn_spec)
{
  std::cerr << "Unknown type for _SingleConnect template\n";
  exit(0);
  
  return 0;
}


template <class T1, class T2>
int NeuralGPU::_ConnectOneToOne(T1 source, T2 target, int n_nodes,
				SynSpec &syn_spec)	       
{
  for (int in=0; in<n_nodes; in++) {
    _SingleConnect<T1, T2>(source, in, target, in, in, syn_spec);
  }

  return 0;
}

template <class T1, class T2>
int NeuralGPU::_ConnectAllToAll
(T1 source, int n_source, T2 target, int n_target, SynSpec &syn_spec)
{
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
  for (int itn=0; itn<n_target; itn++) {
    for (int isn=0; isn<n_source; isn++) {
#ifdef _OPENMP
      omp_set_lock(&(lock[isn]));
#endif
      size_t i_array = (size_t)itn*n_source + isn;
      _SingleConnect<T1, T2>(source, isn, target, itn, i_array, syn_spec);
#ifdef _OPENMP
      omp_unset_lock(&(lock[isn]));
#endif
    }
  }
#ifdef _OPENMP
  delete[] lock;
#endif

  return 0;
}

template <class T1, class T2>
int NeuralGPU::_ConnectFixedTotalNumber
(T1 source, int n_source, T2 target, int n_target, int n_conn,
 SynSpec &syn_spec)
{
  unsigned int *rnd = RandomInt(2*n_conn);
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared)
#endif
  for (int i_conn=0; i_conn<n_conn; i_conn++) {
    int isn = rnd[2*i_conn] % n_source;
    int itn = rnd[2*i_conn+1] % n_target;
#ifdef _OPENMP
    omp_set_lock(&(lock[isn]));
#endif
    _SingleConnect<T1, T2>(source, isn, target, itn, i_conn, syn_spec);
#ifdef _OPENMP
      omp_unset_lock(&(lock[isn]));
#endif
  }
  delete[] rnd;
#ifdef _OPENMP
  delete[] lock;
#endif
  
  return 0;
}




template <class T1, class T2>
int NeuralGPU::_ConnectFixedIndegree
(
 T1 source, int n_source, T2 target, int n_target, int indegree,
 SynSpec &syn_spec
 )
{
  if (indegree>n_source) {
    std::cerr << "Indegree larger than number of source nodes\n";
    exit(0);
  }
  unsigned int *rnd = RandomInt(n_target*indegree);
  std::vector<int> input_array;
  for (int i=0; i<n_source; i++) {
    input_array.push_back(i);
  }
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source];
  for (int i=0; i<n_source; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
  for (int k=0; k<n_target; k++) {
    for (int i=0; i<indegree; i++) {
      int j = i + rnd[k*indegree+i] % (n_source - i);
#ifdef _OPENMP
      omp_set_lock(&(lock[i]));
#endif
      if (j!=i) {
#ifdef _OPENMP
	omp_set_lock(&(lock[j]));
#endif
	std::swap(input_array[i], input_array[j]);
#ifdef _OPENMP
	omp_unset_lock(&(lock[j]));
#endif
      }
      int itn = k;
      int isn = input_array[i];
      size_t i_array = (size_t)k*indegree + i;
      _SingleConnect<T1, T2>(source, isn, target, itn, i_array, syn_spec);
#ifdef _OPENMP
      omp_unset_lock(&(lock[i]));
#endif
    }
  }
  delete[] rnd;
#ifdef _OPENMP
  delete[] lock;
#endif
  
  return 0;
}


template <class T1, class T2>
int NeuralGPU::_RemoteConnect(RemoteNode<T1> source, int n_source,
			      RemoteNode<T2> target, int n_target,
			      ConnSpec &conn_spec, SynSpec &syn_spec)
{
  switch (conn_spec.rule_) {
  case ONE_TO_ONE:
    if (n_source != n_target) {
      std::cerr << "Number of source and target nodes must be equal "
	"for the one-to-one connection rule\n";
      exit(0);
    }
    return _RemoteConnectOneToOne<T1, T2>(source, target, n_source, syn_spec);
    break;
  case ALL_TO_ALL:
    return _RemoteConnectAllToAll<T1, T2>(source, n_source, target, n_target,
				     syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _RemoteConnectFixedTotalNumber<T1, T2>(source, n_source, target,
						  n_target,
						  conn_spec.total_num_,
						  syn_spec);
    break;
  case FIXED_INDEGREE:
    return _RemoteConnectFixedIndegree<T1, T2>(source, n_source, target,
					       n_target, conn_spec.indegree_,
					       syn_spec);
    break;
  case FIXED_OUTDEGREE:
    //return _RemoteConnectFixedOutdegree<T>(source, n_source, target, n_target,
    //conn_spec.outdegree_, syn_spec);
    //break;
  default:
    std::cerr << "Unknown connection rule\n";
    exit(0);
  }
  return 0;
}


template <class T1, class T2>
  int NeuralGPU::_RemoteConnectOneToOne
  (RemoteNode<T1> source, RemoteNode<T2> target, int n_nodes,
   SynSpec &syn_spec)
{
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectOneToOne<T1, T2>(source.i_node_, target.i_node_,
				    n_nodes, syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_nodes];
    int i_new_remote_node;
    if (MpiId() == target.i_host_) {
      i_new_remote_node = net_connection_->connection_.size();
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, source.i_host_);
      //std::vector<ConnGroup> conn;
      //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      BaseNeuron *bn = new BaseNeuron;
      node_vect_.push_back(bn);
      int n_remote_nodes = i_new_remote_node
	- net_connection_->connection_.size();
      //int i_node_0 =
      CreateNodeGroup(n_remote_nodes, 0);	
      //net_connection_->connection_.insert(net_connection_->connection_.end(),
      //				  i_new_remote_node -
      //				  net_connection_->connection_.size(),
      //				  conn);
            
      //NEW, CHECK ///////////
      //InsertNodeGroup(i_new_remote_node
      //		- net_connection_->connection_.size(), 0);
      ///////////////////////
      
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_nodes,
				 source.i_host_);

      for (int in=0; in<n_nodes; in++) {
	int i_remote_node = i_remote_node_arr[in];
	//int i_target_node = i + i_target_node_0;
	// _SingleConnect<int,T2>(i_remote_node, in, target.i_node_, in,
	//		      in, syn_spec);
	_SingleConnect<int,T2>(i_remote_node, 0, target.i_node_, in,
			       in, syn_spec);
	//_SingleConnect<int*,T2>(i_remote_node_arr, in, target.i_node_, in,
	//			in, syn_spec);
      }
    }
    else if (MpiId() == source.i_host_) {
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      for (int in=0; in<n_nodes; in++) {
	//int i_source_node = in + i_source_node_0;
	int i_source_node = source.GetINode(in);
	int i_remote_node = -1;
	for (std::vector<ExternalConnectionNode >::iterator it =
	       connect_mpi_->extern_connection_[i_source_node].begin();
	     it <  connect_mpi_->extern_connection_[i_source_node].end();
	     it++) {
	  if ((*it).target_host_id == target.i_host_) {
	    i_remote_node = (*it).remote_node_id;
	    break;
	  }
	}
	if (i_remote_node == -1) {
	  i_remote_node = i_new_remote_node;
	  i_new_remote_node++;
	  ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	  connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	}
	i_remote_node_arr[in] = i_remote_node;
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_nodes, target.i_host_);
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}


template <class T1, class T2>
  int NeuralGPU::_RemoteConnectFixedIndegree
  (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
   int indegree, SynSpec &syn_spec)
{
  if (MpiId()==source.i_host_ && source.i_host_==target.i_host_) {
    return _ConnectFixedIndegree<T1, T2>(source.i_node_, n_source,
					 target.i_node_, n_target, indegree,
					 syn_spec);
  }
  else if (MpiId()==source.i_host_ || MpiId()==target.i_host_) {
    int *i_remote_node_arr = new int[n_target*indegree];
    int i_new_remote_node;
    if (MpiId() == target.i_host_) {
      i_new_remote_node = net_connection_->connection_.size();
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, source.i_host_);
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, source.i_host_);
      //std::vector<ConnGroup> conn;
      //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      BaseNeuron *bn = new BaseNeuron;
      node_vect_.push_back(bn);
      int n_remote_nodes = i_new_remote_node
	- net_connection_->connection_.size();
      //int i_node_0 =
      CreateNodeGroup(n_remote_nodes, 0);	
      //net_connection_->connection_.insert(net_connection_->connection_.end(),
      //				  i_new_remote_node -
      //				  net_connection_->connection_.size(),
      //				  conn);      
      //NEW, CHECK ///////////
      //InsertNodeGroup(i_new_remote_node
      //		- net_connection_->connection_.size(), 0);
      ///////////////////////
      
      connect_mpi_->MPI_Recv_int(i_remote_node_arr, n_target*indegree,
				 source.i_host_);

      for (int k=0; k<n_target; k++) {
	for (int i=0; i<indegree; i++) {
      	  int i_remote_node = i_remote_node_arr[k*indegree+i];
	  //int i_target_node = k + i_target_node_0;

	  size_t i_array = (size_t)k*indegree + i;
	  //_SingleConnect<int,T2>(i_remote_node, in, target.i_node_, k
	  //				    i_array, syn_spec);
	  _SingleConnect<int,T2>(i_remote_node, 0, target.i_node_, k,
	  			 i_array, syn_spec);
	  
	  //net_connection_->Connect(i_remote_node, i_target_node, i_port, weight, delay);
	}
      }
    }
    else if (MpiId() == source.i_host_) {
      connect_mpi_->MPI_Recv_int(&i_new_remote_node, 1, target.i_host_);
      unsigned int *rnd = RandomInt(n_target*indegree); // check parall. seed problem
      std::vector<int> input_array;
      for (int i=0; i<n_source; i++) {
	//input_array.push_back(i_source_node_0 + i);
	input_array.push_back(source.GetINode(i));
      }
      for (int k=0; k<n_target; k++) {
	for (int i=0; i<indegree; i++) {
	  int j = i + rnd[k*indegree+i] % (n_source - i);
	  if (j!=i) {
	    std::swap(input_array[i], input_array[j]);
	  }
	  int i_source_node = input_array[i];
	  
	  int i_remote_node = -1;
	  for (std::vector<ExternalConnectionNode >::iterator it =
		 connect_mpi_->extern_connection_[i_source_node].begin();
	       it <  connect_mpi_->extern_connection_[i_source_node].end(); it++) {
	    if ((*it).target_host_id == target.i_host_) {
	      i_remote_node = (*it).remote_node_id;
	      break;
	    }
	  }
	  if (i_remote_node == -1) {
	    i_remote_node = i_new_remote_node;
	    i_new_remote_node++;
	    ExternalConnectionNode conn_node = {target.i_host_, i_remote_node};
	    connect_mpi_->extern_connection_[i_source_node].push_back(conn_node);
	  }
	  i_remote_node_arr[k*indegree+i] = i_remote_node;
	}
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_node, 1, target.i_host_);
      connect_mpi_->MPI_Send_int(i_remote_node_arr, n_target*indegree,
				 target.i_host_);
      delete[] rnd;
    }
    delete[] i_remote_node_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}



#endif
