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

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

struct RemoteNeuron
{
  int i_host_;
  int i_neuron_;
};

struct RemoteNeuronPt
{
  int i_host_;
  int *i_neuron_;
};

enum ConnectionRules
  {
   ONE_TO_ONE=0, ALL_TO_ALL, FIXED_TOTAL_NUMBER, FIXED_INDEGREE,
   FIXED_OUTDEGREE, N_CONN_RULE
  };

const std::string conn_rule_name[N_CONN_RULE] =
  {
   "one_to_one", "all_to_all", "fixed_total_number", "fixed_indegree",
   "fixed_outdegree"
};

class ConnSpec
{
  int rule_;
  int total_num_;
  int indegree_;
  int outdegree_;
public:
  ConnSpec();
  ConnSpec(int rule, int degree=0);
  int Init();
  int Init(int rule, int degree=0);
  int SetParam(std::string param_name, int value);
  int GetParam(std::string param_name);
  friend class NeuralGPU;
};

enum SynapseTypes
  {
   STANDARD_SYNAPSE=0, STDP, N_SYNAPSE_TYPE
  };

const std::string synapse_type_name[N_SYNAPSE_TYPE] =
  {
   "standard_synapse", "stdp"
  };

class SynSpec
{
  unsigned char synapse_type_;
  unsigned char receptor_;
 public:
  int weight_distr_;
  float *weight_array_;
  float weight_;
  int delay_distr_;
  float *delay_array_;
  float delay_;
 public:
  SynSpec();
  SynSpec(float weight, float delay);
  SynSpec(int syn_type, float weight, float delay, int receptor=0);
  int Init();
  int Init(float weight, float delay);
  int Init(int syn_type, float weight, float delay, int receptor=0);
  int SetParam(std::string param_name, int value);
  int SetParam(std::string param_name, float value);
  int SetParam(std::string param_name, float *array_pt);
  float GetParam(std::string param_name);
  friend class NeuralGPU;
};
  
template <class T>
int NeuralGPU::_Connect(T source, int n_source, T target, int n_target,
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
    return _ConnectOneToOne<T>(source, target, n_source, syn_spec);
    break;
  case ALL_TO_ALL:
    return _ConnectAllToAll<T>(source, n_source, target, n_target, syn_spec);
    break;
  case FIXED_TOTAL_NUMBER:
    return _ConnectFixedTotalNumber<T>(source, n_source, target, n_target,
				       conn_spec.total_num_, syn_spec);
    break;
  case FIXED_INDEGREE:
    return _ConnectFixedIndegree<T>(source, n_source, target, n_target,
				    conn_spec.indegree_, syn_spec);
    break;
  case FIXED_OUTDEGREE:
    //_ConnectFixedOutdegree<T>(source, n_source, target, n_target,
    //conn_spec.outdegree_, syn_spec);
    //break;
  default:
    std::cerr << "Unknown connection rule\n";
    exit(0);
  }
  return 0;
}

template<class T>
int NeuralGPU::_SingleConnect(T source, int i_source, T target, int i_target,
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
  return _SingleConnect<T>(source, i_source, target, i_target,
			   weight, delay, i_array, syn_spec);
}

template<class T>
int NeuralGPU::_SingleConnect(T source, int i_source, T target, int i_target,
			      float weight, float delay, int i_array,
			      SynSpec &syn_spec)
{
  std::cerr << "Unknown type for _SingleConnect template\n";
  exit(0);
  
  return 0;
}


template <class T>
int NeuralGPU::_ConnectOneToOne(T source, T target, int n_neurons,
				SynSpec &syn_spec)	       
{
  for (int in=0; in<n_neurons; in++) {
    _SingleConnect<T>(source, in, target, in, in, syn_spec);
  }

  return 0;
}

template <class T>
int NeuralGPU::_ConnectAllToAll
(T source, int n_source, T target, int n_target, SynSpec &syn_spec)
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
      _SingleConnect<T>(source, isn, target, itn, i_array, syn_spec);
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

template <class T>
int NeuralGPU::_ConnectFixedTotalNumber
(T source, int n_source, T target, int n_target, int n_conn, SynSpec &syn_spec)
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
    _SingleConnect<T>(source, isn, target, itn, i_conn, syn_spec);
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




template <class T>
int NeuralGPU::_ConnectFixedIndegree
(
 T source, int n_source, T target, int n_target, int indegree, SynSpec &syn_spec
 )
{
  if (indegree>n_source) {
    std::cerr << "Indegree larger than number of source neurons\n";
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
      _SingleConnect<T>(source, isn, target, itn, i_array, syn_spec);
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

#endif
