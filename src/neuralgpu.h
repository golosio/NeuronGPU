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

#ifndef NEURALGPUCLASSH
#define NEURALGPUCLASSH

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "ngpu_exception.h"
#include "node_group.h"
#include "base_neuron.h"
#include "connect_spec.h"
#include "connect.h"

class PoissonGenerator;
class Multimeter;
class NetConnection;
class ConnectMpi;
struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;
//struct RemoteNode;
//struct RemoteNodePt;
class ConnSpec;
class SynSpec;

class Sequence
{
 public:
  int i0;
  int n;
  
 Sequence(int i0=0, int n=0) : i0(i0), n(n) {}
  
  inline int operator[](int i) {
    if (i<0) {
      throw ngpu_exception("Sequence index cannot be negative");
    }
    if (i>=n) {
      throw ngpu_exception("Sequence index out of range");
    }
    return i0 + i;
  }

  inline Sequence Subseq(int first, int last) {
    if (first<0 || first>last) {
      throw ngpu_exception("Sequence subset range error");
    }
    if (last>=n) {
      throw ngpu_exception("Sequence subset out of range");
    }
    return Sequence(i0 + first, last - first + 1);
  }

  // https://stackoverflow.com/questions/18625223
  inline std::vector<int> ToVector() {
    std::vector<int> v;
    v.reserve(n);
    std::generate_n(std::back_inserter(v), n, [&](){ return i0 + v.size(); });
    return v;
  }
};

typedef Sequence NodeSeq;

enum {ON_EXCEPTION_EXIT=0, ON_EXCEPTION_HANDLE};

class NeuralGPU
{
  float time_resolution_; // time resolution in ms
  curandGenerator_t *random_generator_;
  unsigned long long kernel_seed_;
  bool calibrate_flag_; // becomes true after calibration
  bool mpi_flag_; // true if MPI is initialized

  PoissonGenerator *poiss_generator_;
  Multimeter *multimeter_;
  std::vector<BaseNeuron*> node_vect_; // -> node_group_vect
  
  NetConnection *net_connection_;
  ConnectMpi *connect_mpi_;

  std::vector<signed char> node_group_map_;
  signed char *d_node_group_map_;


  int max_spike_buffer_size_;
  int max_spike_num_;
  int max_spike_per_host_;

  float t_min_;
  float neural_time_; // Neural activity time
  float sim_time_; // Simulation time in ms
  int n_poiss_node_;

  double start_real_time_;
  double build_real_time_;
  double end_real_time_;

  bool error_flag_;
  std::string error_message_;
  unsigned char error_code_;
  int on_exception_;
  
  int CreateNodeGroup(int n_neuron, int n_port);
  int CheckUncalibrated(std::string message);
  double *InitGetSpikeArray(int n_node, int n_port);
  int NodeGroupArrayInit();
  int ClearGetSpikeArrays();
  int FreeGetSpikeArrays();
  int FreeNodeGroupMap();


  template <class T1, class T2>
    int _Connect(T1 source, int n_source, T2 target, int n_target,
		 ConnSpec &conn_spec, SynSpec &syn_spec);
  
  template<class T1, class T2>
    int _SingleConnect(T1 source, int i_source, T2 target, int i_target,
		       int i_array, SynSpec &syn_spec);
  template<class T1, class T2>
    int _SingleConnect(T1 source, int i_source, T2 target, int i_target,
		       float weight, float delay, int i_array,
		       SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectOneToOne(T1 source, T2 target, int n_node, SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectAllToAll
    (T1 source, int n_source, T2 target, int n_target, SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectFixedTotalNumber
    (T1 source, int n_source, T2 target, int n_target, int n_conn,
     SynSpec &syn_spec);

  template <class T1, class T2>
    int _ConnectFixedIndegree
    (
     T1 source, int n_source, T2 target, int n_target, int indegree,
     SynSpec &syn_spec
     );

  template <class T1, class T2>
    int _ConnectFixedOutdegree
    (
     T1 source, int n_source, T2 target, int n_target, int outdegree,
     SynSpec &syn_spec
     );

  template <class T1, class T2>
    int _RemoteConnect(RemoteNode<T1> source, int n_source,
		       RemoteNode<T2> target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec);
  
  template <class T1, class T2>
    int _RemoteConnectOneToOne
    (RemoteNode<T1> source, RemoteNode<T2> target, int n_node,
     SynSpec &syn_spec);
  
  template <class T1, class T2>
    int _RemoteConnectAllToAll
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectFixedTotalNumber
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int n_conn, SynSpec &syn_spec);
  
  template <class T1, class T2>
    int _RemoteConnectFixedIndegree
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int indegree, SynSpec &syn_spec);

  template <class T1, class T2>
    int _RemoteConnectFixedOutdegree
    (RemoteNode<T1> source, int n_source, RemoteNode<T2> target, int n_target,
     int outdegree, SynSpec &syn_spec);

    
 public:
  NeuralGPU();

  ~NeuralGPU();

  int SetRandomSeed(unsigned long long seed);

  int SetTimeResolution(float time_res);
  
  inline float GetTimeResolution() {
    return time_resolution_;
  }

  inline int SetSimTime(float sim_time) {
    sim_time_ = sim_time;
    return 0;
  }

  inline float GetSimTime() {
    return sim_time_;
  }

  int SetMaxSpikeBufferSize(int max_size);
  int GetMaxSpikeBufferSize();
  NodeSeq Create(std::string model_name, int n_neuron=1, int n_port=1);
  NodeSeq CreatePoissonGenerator(int n_node, float rate);
  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_node_arr, int n_node);  
  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_node_arr, int *i_port_arr, int n_node);
  std::vector<std::vector<float>> *GetRecordData(int i_record);

  int SetNeuronParam(int i_node, int n_neuron, std::string param_name,
		     float val);

  int SetNeuronParam(int *i_node, int n_neuron, std::string param_name,
		     float val);

  int SetNeuronParam(int i_node, int n_neuron, std::string param_name,
		     float *param, int array_size);

  int SetNeuronParam(int *i_node, int n_neuron, std::string param_name,
		     float *param, int array_size);

  int SetNeuronParam(NodeSeq nodes, std::string param_name, float val) {
    return SetNeuronParam(nodes.i0, nodes.n, param_name, val);
  }

  int SetNeuronParam(NodeSeq nodes, std::string param_name, float *param,
		      int array_size) {
    return SetNeuronParam(nodes.i0, nodes.n, param_name, param, array_size);
  }
  
  int SetNeuronParam(std::vector<int> nodes, std::string param_name,
		     float val) {
    return SetNeuronParam(nodes.data(), nodes.size(), param_name, val);
  }

  int SetNeuronParam(std::vector<int> nodes, std::string param_name,
		     float *param, int array_size) {
    return SetNeuronParam(nodes.data(), nodes.size(), param_name, param,
			  array_size);
  }

  int SetNeuronVar(int i_node, int n_neuron, std::string var_name,
		     float val);

  int SetNeuronVar(int *i_node, int n_neuron, std::string var_name,
		     float val);

  int SetNeuronVar(int i_node, int n_neuron, std::string var_name,
		     float *var, int array_size);

  int SetNeuronVar(int *i_node, int n_neuron, std::string var_name,
		     float *var, int array_size);

  int SetNeuronVar(NodeSeq nodes, std::string var_name, float val) {
    return SetNeuronVar(nodes.i0, nodes.n, var_name, val);
  }

  int SetNeuronVar(NodeSeq nodes, std::string var_name, float *var,
		      int array_size) {
    return SetNeuronVar(nodes.i0, nodes.n, var_name, var, array_size);
  }
  
  int SetNeuronVar(std::vector<int> nodes, std::string var_name,
		     float val) {
    return SetNeuronVar(nodes.data(), nodes.size(), var_name, val);
  }

  int SetNeuronVar(std::vector<int> nodes, std::string var_name,
		     float *var, int array_size) {
    return SetNeuronVar(nodes.data(), nodes.size(), var_name, var,
			  array_size);
  }

  int GetNeuronParamSize(int i_node, std::string param_name);

  int GetNeuronVarSize(int i_node, std::string var_name);

  float *GetNeuronParam(int i_node, int n_neuron, std::string param_name);

  float *GetNeuronParam(int *i_node, int n_neuron, std::string param_name);

  float *GetNeuronParam(NodeSeq nodes, std::string param_name) {
    return GetNeuronParam(nodes.i0, nodes.n, param_name);
  }
  
  float *GetNeuronParam(std::vector<int> nodes, std::string param_name) {
    return GetNeuronParam(nodes.data(), nodes.size(), param_name);
  }

  float *GetArrayParam(int i_node, std::string param_name);
  
  float *GetNeuronVar(int i_node, int n_neuron, std::string var_name);

  float *GetNeuronVar(int *i_node, int n_neuron, std::string var_name);

  float *GetNeuronVar(NodeSeq nodes, std::string var_name) {
    return GetNeuronVar(nodes.i0, nodes.n, var_name);
  }
  
  float *GetNeuronVar(std::vector<int> nodes, std::string var_name) {
    return GetNeuronVar(nodes.data(), nodes.size(), var_name);
  }

  float *GetArrayVar(int i_node, std::string param_name);
  
  int GetNodeSequenceOffset(int i_node, int n_node, int &i_group);

  std::vector<int> GetNodeArrayWithOffset(int *i_node, int n_node,
					  int &i_group);

  int IsNeuronScalParam(int i_node, std::string param_name);

  int IsNeuronPortParam(int i_node, std::string param_name);

  int IsNeuronArrayParam(int i_node, std::string param_name);

  int IsNeuronScalVar(int i_node, std::string var_name);

  int IsNeuronPortVar(int i_node, std::string var_name);

  int IsNeuronArrayVar(int i_node, std::string var_name);
  
  int SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
			float *spike_height);

  int Calibrate();
  int Simulate(float sim_time=1000.0);

  int ConnectMpiInit(int argc, char *argv[]);

  int MpiId();

  int MpiNp();

  int ProcMaster();

  int MpiFinalize();

  void SetErrorFlag(bool error_flag) {error_flag_ = error_flag;}
  
  void SetErrorMessage(std::string error_message) { error_message_
      = error_message; }

  void SetErrorCode(unsigned char error_code) {error_code_ = error_code;}

  void SetOnException(int on_exception) {on_exception_ = on_exception;}

  bool GetErrorFlag() {return error_flag_;}

  char *GetErrorMessage() {return &error_message_[0];}

  unsigned char GetErrorCode() {return error_code_;}

  int OnException() {return on_exception_;}

  unsigned int *RandomInt(size_t n);
  
  float *RandomUniform(size_t n);

  float *RandomNormal(size_t n, float mean, float stddev);

  float *RandomNormalClipped(size_t n, float mean, float stddev, float vmin,
			     float vmax);  

  int Connect
    (
     int i_source_node, int i_target_node, unsigned char i_port,
     float weight, float delay
     );

  int Connect(int i_source, int n_source, int i_target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int i_source, int n_source, int* target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int* source, int n_source, int i_target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(int* source, int n_source, int* target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(NodeSeq source, NodeSeq target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(NodeSeq source, std::vector<int> target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(std::vector<int> source, NodeSeq target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int Connect(std::vector<int> source, std::vector<int> target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int i_source, int n_source,
		    int i_target_host, int i_target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int i_source, int n_source,
		    int i_target_host, int* target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int* source, int n_source,
		    int i_target_host, int i_target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, int* source, int n_source,
		    int i_target_host, int* target, int n_target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, NodeSeq source,
		    int i_target_host, NodeSeq target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, NodeSeq source,
		    int i_target_host, std::vector<int> target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, std::vector<int> source,
		    int i_target_host, NodeSeq target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int RemoteConnect(int i_source_host, std::vector<int> source,
		    int i_target_host, std::vector<int> target,
		    ConnSpec &conn_spec, SynSpec &syn_spec);

  int BuildDirectConnections();

  std::vector<std::string> GetScalVarNames(int i_node);

  int GetNScalVar(int i_node);
  
  std::vector<std::string> GetPortVarNames(int i_node);

  int GetNPortVar(int i_node);
  
  std::vector<std::string> GetScalParamNames(int i_node);

  int GetNScalParam(int i_node);
  
  std::vector<std::string> GetPortParamNames(int i_node);

  int GetNPortParam(int i_node);
  
  std::vector<std::string> GetArrayParamNames(int i_node);

  int GetNArrayParam(int i_node);

  std::vector<std::string> GetArrayVarNames(int i_node);

  int GetNArrayVar(int i_node);

  ConnectionStatus GetConnectionStatus(ConnectionId conn_id);
  
  std::vector<ConnectionStatus> GetConnectionStatus(std::vector<ConnectionId>
						    &conn_id_vect);

  std::vector<ConnectionId> GetConnections(int i_source, int n_source,
					   int i_target, int n_target,
					   int syn_type=0);

  std::vector<ConnectionId> GetConnections(int *i_source, int n_source,
					   int i_target, int n_target,
					   int syn_type=0);

  std::vector<ConnectionId> GetConnections(int i_source, int n_source,
					   int *i_target, int n_target,
					   int syn_type=0);

  std::vector<ConnectionId> GetConnections(int *i_source, int n_source,
					   int *i_target, int n_target,
					   int syn_type=0);
    
  std::vector<ConnectionId> GetConnections(NodeSeq source, NodeSeq target,
					   int syn_type=0);

  std::vector<ConnectionId> GetConnections(std::vector<int> source,
					   NodeSeq target, int syn_type=0);

  std::vector<ConnectionId> GetConnections(NodeSeq source,
					   std::vector<int> target,
					   int syn_type=0);

  std::vector<ConnectionId> GetConnections(std::vector<int> source,
					   std::vector<int> target,
					   int syn_type=0);

};



#endif
