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

#include "neuron_group.h"
#include "base_neuron.h"

class PoissonGenerator;
class SpikeGenerator;
class Multimeter;
class NetConnection;
class ConnectMpi;
struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;
struct RemoteNeuron;
struct RemoteNeuronPt;
class ConnSpec;
class SynSpec;

class Sequence
{
 public:
    int i0;
    int n;

    Sequence(int i0=0, int n=0) : i0(i0), n(n) {}
    
    inline int operator[](int i) {return i0 + i;} 
    inline Sequence Subset(int first, int last) {
      if (first<0 || last>=n) {
      std::cerr << "Sequence subset out of range\n";
      exit(0);
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

typedef Sequence Nodes;

class NeuralGPU
{
  float time_resolution_; // time resolution in ms
  curandGenerator_t *random_generator_;
  bool calibrate_flag_; // becomes true after calibration
  bool mpi_flag_; // true if MPI is initialized

  PoissonGenerator *poiss_generator_;
  SpikeGenerator *spike_generator_;
  Multimeter *multimeter_;
  std::vector<BaseNeuron*> neuron_vect_;
  
  NetConnection *net_connection_;
  ConnectMpi *connect_mpi_;

  std::vector<NeuronGroup> neuron_group_vect_;
  std::vector<signed char> neuron_group_map_;
  signed char *d_neuron_group_map_;


  int max_spike_buffer_size_;
  int max_spike_num_;
  int max_spike_per_host_;

  float t_min_;
  float neural_time_; // Neural activity time
  float sim_time_; // Simulation time in ms
  int n_neurons_;
  int n_poiss_nodes_;
  int n_spike_gen_nodes_;

  double start_real_time_;
  double build_real_time_;
  double end_real_time_;
  
  int CreateNeuron(int n_neurons, int n_receptors);
  int CheckUncalibrated(std::string message);
  int InsertNeuronGroup(int n_neurons, int n_receptors);
  int NeuronGroupArrayInit();
  int ClearGetSpikeArrays();
  int FreeGetSpikeArrays();
  int FreeNeuronGroupMap();


  template <class T>
  int _Connect(T source, int n_source, T target, int n_target,
	       ConnSpec &conn_spec, SynSpec &syn_spec);
  
  template<class T>
  int _SingleConnect(T source, int i_source, T target, int i_target,
		     int i_array, SynSpec &syn_spec);
  template<class T>
  int _SingleConnect(T source, int i_source, T target, int i_target,
		     float weight, float delay, int i_array, SynSpec &syn_spec);

  template <class T>
  int _ConnectOneToOne(T source, T target, int n_neurons, SynSpec &syn_spec);
  template <class T>
  int _ConnectAllToAll
  (T source, int n_source, T target, int n_target, SynSpec &syn_spec);

  template <class T>
  int _ConnectFixedTotalNumber
  (T source, int n_source, T target, int n_target, int n_conn,
   SynSpec &syn_spec);

  template <class T>
  int _ConnectFixedIndegree
  (
   T source, int n_source, T target, int n_target, int indegree,
   SynSpec &syn_spec
   );



  
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
  Nodes CreateNeuron(std::string model_name, int n_neurons, int n_receptors);
  Nodes CreatePoissonGenerator(int n_nodes, float rate);
  Nodes CreateSpikeGenerator(int n_nodes);
  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_neuron_arr, int n_neurons);  
  int CreateRecord(std::string file_name, std::string *var_name_arr,
		   int *i_neuron_arr, int *i_receptor_arr, int n_neurons);
  std::vector<std::vector<float>> *GetRecordData(int i_record);

  int SetNeuronParams(std::string param_name, int i_node, int n_neurons,
		      float val);

  int SetNeuronParams(std::string param_name, int i_node, int n_neurons,
			  float *params, int vect_size);

  int SetNeuronParams(std::string param_name, Nodes nodes, float val) {
    return SetNeuronParams(param_name, nodes.i0, nodes.n, val);
  }

  int SetNeuronParams(std::string param_name, Nodes nodes, float *params,
		      int vect_size) {
    return SetNeuronParams(param_name, nodes.i0, nodes.n, params, vect_size);
  }
  int SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
			float *spike_height);

  int Calibrate();
  int Simulate();

  int ConnectMpiInit(int argc, char *argv[]);

  int MpiId();

  int MpiNp();

  int ProcMaster();

  int MpiFinalize();
  
  unsigned int *RandomInt(size_t n);
  
  float *RandomUniform(size_t n);

  float *RandomNormal(size_t n, float mean, float stddev);

  float *RandomNormalClipped(size_t n, float mean, float stddev, float vmin,
			     float vmax);  

  int Connect
    (
     int i_source_neuron, int i_target_neuron, unsigned char i_port,
     float weight, float delay
     );

  int ConnectOneToOne
    (
     int i_source_neuron_0, int i_target_neuron_0, int n_neurons,
     unsigned char i_port, float weight, float delay
     );

  int ConnectAllToAll
    (
     int i_source_neuron_0, int n_source_neurons,
     int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay
     );
  
  int ConnectFixedIndegree
    (
     int i_source_neuron_0, int n_source_neurons,
     int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay, int indegree
     );

  int ConnectFixedIndegreeArray
    (
     int i_source_neuron_0, int n_source_neurons,
     int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float *weight_arr, float *delay_arr, int indegree
     );
  
  int ConnectFixedTotalNumberArray(int i_source_neuron_0, int n_source_neurons,
				   int i_target_neuron_0, int n_target_neurons,
				   unsigned char i_port, float *weight_arr,
				   float *delay_arr, int n_conn);

  int RemoteConnect(int i_source_host, int i_source_neuron,
		    int i_target_host, int i_target_neuron,
		    unsigned char i_port, float weight, float delay);
  
  int RemoteConnectOneToOne
    (
     int i_source_host, int i_source_neuron_0,
     int i_target_host, int i_target_neuron_0, int n_neurons,
     unsigned char i_port, float weight, float delay
     );

  int RemoteConnectAllToAll
    (
     int i_source_host, int i_source_neuron_0, int n_source_neurons,
     int i_target_host, int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay
     );
  
  int RemoteConnectFixedIndegree
    (
     int i_source_host, int i_source_neuron_0, int n_source_neurons,
     int i_target_host, int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay, int indegree
     );
  
  int Connect(int i_source, int n_source, int i_target, int n_target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);
  
  int Connect(Nodes source, Nodes target,
	      ConnSpec &conn_spec, SynSpec &syn_spec);


};



#endif
