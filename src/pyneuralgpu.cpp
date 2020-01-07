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

#include <iostream>
#include <cstdlib>
#include <string>

#include "neuralgpu.h"
#include "pyneuralgpu.h"

extern "C" {
  static NeuralGPU *NeuralGPU_instance = NULL;

  void checkNeuralGPUInstance() {
    if (NeuralGPU_instance == NULL) {
      NeuralGPU_instance = new NeuralGPU();
    }
  }

  int NeuralGPU_SetRandomSeed(unsigned long long seed)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->SetRandomSeed(seed);
  }

  int NeuralGPU_SetTimeResolution(float time_res)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->SetTimeResolution(time_res);
  }

  float NeuralGPU_GetTimeResolution()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->GetTimeResolution();
  }

  int NeuralGPU_SetMaxSpikeBufferSize(int max_size)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->SetMaxSpikeBufferSize(max_size);
  }

  int NeuralGPU_GetMaxSpikeBufferSize()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->GetMaxSpikeBufferSize();
  }

  int NeuralGPU_CreateNeuron(char *model_name, int n_neurons, int n_receptors)
  {
    checkNeuralGPUInstance();
    std::string model_name_str = std::string(model_name);
    return NeuralGPU_instance->CreateNeuron(model_name_str, n_neurons,
					    n_receptors);
  }

  int NeuralGPU_CreatePoissonGenerator(int n_nodes, float rate)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->CreatePoissonGenerator(n_nodes, rate);
  }
  
  int NeuralGPU_CreateSpikeGenerator(int n_nodes)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->CreateSpikeGenerator(n_nodes);
  }
  
  int NeuralGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_neuron_arr, int *i_receptor_arr,
			     int n_neurons)
  {
    checkNeuralGPUInstance();
    std::string file_name_str = std::string(file_name);
    std::vector<std::string> var_name_vect;
    for (int i=0; i<n_neurons; i++) {
      std::string var_name = std::string(var_name_arr[i]);
      var_name_vect.push_back(var_name);
    }
    return NeuralGPU_instance->CreateRecord
      (file_name_str, var_name_vect.data(), i_neuron_arr, i_receptor_arr,
       n_neurons);		       
  }
  
  int NeuralGPU_GetRecordDataRows(int i_record)
  {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);

    return data_vect_pt->size();
  }
  
  int NeuralGPU_GetRecordDataColumns(int i_record)
  {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);
    
    return data_vect_pt->at(0).size();
  }

  float **NeuralGPU_GetRecordData(int i_record)
  {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);
    int nr = data_vect_pt->size();
    float **data_arr = new float*[nr];
    for (int i=0; i<nr; i++) {
      data_arr[i] = data_vect_pt->at(i).data();
    }
    
    return data_arr; 
  }

  int NeuralGPU_SetNeuronParams(char *param_name, int i_node, int n_neurons,
				float val)
  {
    checkNeuralGPUInstance();
    std::string param_name_str = std::string(param_name);
    return NeuralGPU_instance->SetNeuronParams(param_name_str, i_node,
					       n_neurons, val);
  }

  int NeuralGPU_SetNeuronVectParams(char *param_name, int i_node,
				    int n_neurons, float *params,
				    int vect_size)
  {
    checkNeuralGPUInstance();
    std::string param_name_str = std::string(param_name);    
    return NeuralGPU_instance->SetNeuronVectParams(param_name_str, i_node,
						   n_neurons, params,
						   vect_size);
  }

  int NeuralGPU_SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
			float *spike_height)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->SetSpikeGenerator(i_node, n_spikes, spike_time,
						 spike_height);
  }

  int NeuralGPU_Calibrate()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->Calibrate();
  }

  int NeuralGPU_Simulate()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->Simulate();
  }

  int NeuralGPU_ConnectMpiInit(int argc, char *argv[])
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->ConnectMpiInit(argc, argv);
  }

  int NeuralGPU_MpiId()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->MpiId();
  }

  int NeuralGPU_MpiNp()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->MpiNp();
  }
  int NeuralGPU_ProcMaster()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->ProcMaster();
  }

  int NeuralGPU_MpiFinalize()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->MpiFinalize();
  }

  unsigned int *NeuralGPU_RandomInt(size_t n)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->RandomInt(n);
  }
  
  float *NeuralGPU_RandomUniform(size_t n)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->RandomUniform(n);
  }
  
  float *NeuralGPU_RandomNormal(size_t n, float mean, float stddev)
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->RandomNormal(n, mean, stddev);
  }
  
  float *NeuralGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax)
  {
    checkNeuralGPUInstance();
    float *arr = NeuralGPU_instance->RandomNormalClipped(n, mean, stddev, vmin,
							 vmax);
    return arr;
  }
  
  int NeuralGPU_Connect(int i_source_neuron, int i_target_neuron,
			unsigned char i_port, float weight, float delay)
  {
    return NeuralGPU_instance->Connect(i_source_neuron, i_target_neuron,
				       i_port, weight, delay);
  }

  int NeuralGPU_ConnectOneToOne
  (
   int i_source_neuron_0, int i_target_neuron_0, int n_neurons,
   unsigned char i_port, float weight, float delay
   )
  {
    return NeuralGPU_instance->ConnectOneToOne
      (i_source_neuron_0, i_target_neuron_0, n_neurons, i_port, weight, delay);
  }

  int NeuralGPU_ConnectAllToAll
  (
   int i_source_neuron_0, int n_source_neurons,
   int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float weight, float delay
   )
  {
    return NeuralGPU_instance->ConnectAllToAll
      (
       i_source_neuron_0, n_source_neurons, i_target_neuron_0,
       n_target_neurons, i_port, weight, delay
       );
  }
  
  int NeuralGPU_ConnectFixedIndegree
  (
   int i_source_neuron_0, int n_source_neurons,
   int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float weight, float delay, int indegree
   )
  {
    return NeuralGPU_instance->ConnectFixedIndegree
      (
       i_source_neuron_0, n_source_neurons, i_target_neuron_0,
       n_target_neurons, i_port, weight, delay, indegree
     );
  }

  int NeuralGPU_ConnectFixedIndegreeArray
  (
   int i_source_neuron_0, int n_source_neurons,
   int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float *weight_arr, float *delay_arr, int indegree
   )
  {
    return NeuralGPU_instance->ConnectFixedIndegreeArray
      (
       i_source_neuron_0, n_source_neurons, i_target_neuron_0,
       n_target_neurons, i_port, weight_arr, delay_arr, indegree
       );
  }
  
  int NeuralGPU_ConnectFixedTotalNumberArray
  (
   int i_source_neuron_0, int n_source_neurons,
   int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float *weight_arr,
   float *delay_arr, int n_conn
   )
  {
    return NeuralGPU_instance->ConnectFixedTotalNumberArray
      (
       i_source_neuron_0, n_source_neurons, i_target_neuron_0,
       n_target_neurons, i_port, weight_arr, delay_arr, n_conn
       );
  }

  int NeuralGPU_RemoteConnect
  (
   int i_source_host, int i_source_neuron,
   int i_target_host, int i_target_neuron,
   unsigned char i_port, float weight, float delay
   )
  {
    return NeuralGPU_instance->RemoteConnect
      (
       i_source_host, i_source_neuron, i_target_host, i_target_neuron,
       i_port, weight, delay
       );
  }
  
  int NeuralGPU_RemoteConnectOneToOne
  (
   int i_source_host, int i_source_neuron_0,
   int i_target_host, int i_target_neuron_0, int n_neurons,
   unsigned char i_port, float weight, float delay
   )
  {
    return NeuralGPU_instance->RemoteConnectOneToOne
      (
       i_source_host, i_source_neuron_0, i_target_host, i_target_neuron_0,
       n_neurons, i_port, weight, delay
       );
  }

  int NeuralGPU_RemoteConnectAllToAll
  (
   int i_source_host, int i_source_neuron_0, int n_source_neurons,
   int i_target_host, int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float weight, float delay
   )
  {
    return NeuralGPU_instance->RemoteConnectAllToAll
  (
   i_source_host, i_source_neuron_0, n_source_neurons, i_target_host,
   i_target_neuron_0, n_target_neurons, i_port, weight, delay
   );
  }
  
  int NeuralGPU_RemoteConnectFixedIndegree
  (
   int i_source_host, int i_source_neuron_0, int n_source_neurons,
   int i_target_host, int i_target_neuron_0, int n_target_neurons,
   unsigned char i_port, float weight, float delay, int indegree
   )
  {
    return NeuralGPU_instance->RemoteConnectFixedIndegree
      (
       i_source_host, i_source_neuron_0, n_source_neurons, i_target_host,
       i_target_neuron_0, n_target_neurons, i_port, weight, delay, indegree
       );
  }

}

