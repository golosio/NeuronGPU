/*
Copyright (C) 2019 Bruno Golosio
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

#ifndef NEURAL_GPU_H
#define NEURAL_GPU_H

#include <vector>
#include <string>

#include "neuron_group.h"
#include "base_neuron.h"
//#include "connect.h"
//#include "connect_mpi.h"
//#include "poisson.h"
//#include "aeif.h"
//#include "spike_generator.h"
//#include "multimeter.h"

//class curandGenerator_t;
struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;
class PoissonGenerator;
class SpikeGenerator;
class Multimeter;
class AEIF;
class NetConnection;
class ConnectMpi;

class NeuralGPU
{
  float time_resolution_; // time resolution in ms
  curandGenerator_t *random_generator_;
 public:
  PoissonGenerator *poiss_generator_;
  SpikeGenerator *spike_generator_;
  Multimeter *multimeter_;
  std::vector<BaseNeuron*> neuron_vect_;
  
  NetConnection *net_connection_;
  ConnectMpi *connect_mpi_;

  std::vector<NeuronGroup> neuron_group_vect_;
  std::vector<signed char> neuron_group_map_;
  signed char *d_neuron_group_map_;
  
  int max_spike_buffer_num_;
  int max_spike_num_;
  int max_spike_per_host_;

  float t_min_;
  float sim_time_; // Simulation time in ms
  int n_neurons_;
  int n_poiss_nodes_;
  int n_spike_gen_nodes_;

  double start_real_time_;
  double build_real_time_;
  double end_real_time_;
    
  NeuralGPU();

  ~NeuralGPU();

  int SetRandomSeed(unsigned long long seed);

  int SetTimeResolution(float time_res);

  inline float GetTimeResolution() {
    return time_resolution_;
  }

  int CreateNeuron(int n_neurons, int n_receptors);
  int CreatePoissonGenerator(int n_nodes, float rate);
  int CreateSpikeGenerator(int n_nodes);
  int CreateRecord(std::string file_name, std::string var_name, int *i_neurons,
		   int n_neurons);
  
  int Simulate();

  int ConnectFixedIndegree
    (
     int i_source_neuron_0, int n_source_neurons,
     int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay, int indegree
     );

  int RemoteConnectFixedIndegree
    (
     int i_source_host, int i_source_neuron_0, int n_source_neurons,
     int i_target_host, int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay, int indegree
     );
  
  int ConnectAllToAll
    (
     int i_source_neuron_0, int n_source_neurons,
     int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay
     );

  int RemoteConnectAllToAll
    (
     int i_source_host, int i_source_neuron_0, int n_source_neurons,
     int i_target_host, int i_target_neuron_0, int n_target_neurons,
     unsigned char i_port, float weight, float delay
     );

  int ConnectOneToOne
    (
     int i_source_neuron_0, int i_target_neuron_0, int n_neurons,
     unsigned char i_port, float weight, float delay
     );

  int RemoteConnectOneToOne
    (
     int i_source_host, int i_source_neuron_0,
     int i_target_host, int i_target_neuron_0, int n_neurons,
     unsigned char i_port, float weight, float delay
     );
  
  int RemoteConnect(int i_source_host, int i_source_neuron,
		    int i_target_host, int i_target_neuron,
		    unsigned char i_port, float weight, float delay);
    
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

  int SetNeuronParams(std::string param_name, int i_node, int n_neurons,
		      float val);

  int SetNeuronVectParams(std::string param_name, int i_node, int n_neurons,
			  float *params, int vect_size);

  int ConnectMpiInit(int argc, char *argv[]);

  int MpiId();

  int MpiNp();

  int ProcMaster();

  int MpiFinalize();

  int SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
			float *spike_height);
  
  unsigned int *RandomInt(size_t n);
  
  float *RandomUniform(size_t n);

  float *RandomNormal(size_t n, float mean, float stddev);

  float *RandomNormalClipped(size_t n, float mean, float stddev, float vmin,
			     float vmax);  

  int InsertNeuronGroup(int n_neurons, int n_receptors);
  int NeuronGroupArrayInit();
  int ClearGetSpikeArrays();
  int FreeGetSpikeArrays();
  int FreeNeuronGroupMap();
};

#endif
