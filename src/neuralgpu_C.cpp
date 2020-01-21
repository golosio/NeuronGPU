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
#include "neuralgpu_C.h"
#include "propagate_error.h"

extern "C" {
  static NeuralGPU *NeuralGPU_instance = NULL;
  ConnSpec ConnSpec_instance;
  SynSpec SynSpec_instance;

  void checkNeuralGPUInstance() {
    if (NeuralGPU_instance == NULL) {
      NeuralGPU_instance = new NeuralGPU();
    }
  }
  
  char *NeuralGPU_GetErrorMessage()
  {
    checkNeuralGPUInstance();
    char *cstr = NeuralGPU_instance->GetErrorMessage();
    return cstr;
  }

  unsigned char NeuralGPU_GetErrorCode()
  {
    checkNeuralGPUInstance();
    return NeuralGPU_instance->GetErrorCode();
  }

  void NeuralGPU_SetOnException(int on_exception)
  {
    checkNeuralGPUInstance();
    NeuralGPU_instance->SetOnException(on_exception);
  }

  unsigned int *RandomInt(size_t n);
  
  int NeuralGPU_SetRandomSeed(unsigned long long seed)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->SetRandomSeed(seed);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetTimeResolution(float time_res)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->SetTimeResolution(time_res);
  } END_ERR_PROP return ret; }

  float NeuralGPU_GetTimeResolution()
  { float ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->GetTimeResolution();
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetMaxSpikeBufferSize(int max_size)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->SetMaxSpikeBufferSize(max_size);
  } END_ERR_PROP return ret; }

  int NeuralGPU_GetMaxSpikeBufferSize()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->GetMaxSpikeBufferSize();
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetSimTime(float sim_time)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->SetSimTime(sim_time);
  } END_ERR_PROP return ret; }

  int NeuralGPU_Create(char *model_name, int n_neurons, int n_ports)
  { int ret; BEGIN_ERR_PROP {
    std::string model_name_str = std::string(model_name);
    NodeSeq neur = NeuralGPU_instance->Create(model_name_str, n_neurons,
						    n_ports);
    ret = neur[0];
  } END_ERR_PROP return ret; }

  int NeuralGPU_CreatePoissonGenerator(int n_nodes, float rate)
  { int ret; BEGIN_ERR_PROP {
    NodeSeq pg = NeuralGPU_instance->CreatePoissonGenerator(n_nodes, rate);

    ret = pg[0];
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *i_port_arr,
			     int n_nodes)
  { int ret; BEGIN_ERR_PROP {
    std::string file_name_str = std::string(file_name);
    std::vector<std::string> var_name_vect;
    for (int i=0; i<n_nodes; i++) {
      std::string var_name = std::string(var_name_arr[i]);
      var_name_vect.push_back(var_name);
    }
    ret = NeuralGPU_instance->CreateRecord
      (file_name_str, var_name_vect.data(), i_node_arr, i_port_arr,
       n_nodes);		       
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_GetRecordDataRows(int i_record)
  { int ret; BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);

    ret = data_vect_pt->size();
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_GetRecordDataColumns(int i_record)
  { int ret; BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);
    
    ret = data_vect_pt->at(0).size();
  } END_ERR_PROP return ret; }

  float **NeuralGPU_GetRecordData(int i_record)
  { float **ret; BEGIN_ERR_PROP {
    std::vector<std::vector<float>> *data_vect_pt
      = NeuralGPU_instance->GetRecordData(i_record);
    int nr = data_vect_pt->size();
    ret = new float*[nr];
    for (int i=0; i<nr; i++) {
      ret[i] = data_vect_pt->at(i).data();
    }
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronScalParam(int i_node, int n_neurons, char *param_name,
				   float val)
  { int ret; BEGIN_ERR_PROP {
    
    std::string param_name_str = std::string(param_name);
    ret = NeuralGPU_instance->SetNeuronParam(i_node, n_neurons,
					     param_name_str, val);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronArrayParam(int i_node, int n_neurons,
				    char *param_name, float *params,
				    int array_size)
  { int ret; BEGIN_ERR_PROP {
      std::string param_name_str = std::string(param_name);    
      ret = NeuralGPU_instance->SetNeuronParam(i_node, n_neurons,
					       param_name_str, params,
					       array_size);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronPtScalParam(int *i_node, int n_neurons,
				     char *param_name,float val)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = NeuralGPU_instance->SetNeuronParam(i_node, n_neurons,
					     param_name_str, val);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronPtArrayParam(int *i_node, int n_neurons,
				     char *param_name, float *params,
				     int array_size)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);    
    ret = NeuralGPU_instance->SetNeuronParam(i_node, n_neurons,
					     param_name_str, params,
					     array_size);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronScalParam(int i_node, char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NeuralGPU_instance->IsNeuronScalParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronPortParam(int i_node, char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NeuralGPU_instance->IsNeuronPortParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronArrayParam(int i_node, char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    
    ret = NeuralGPU_instance->IsNeuronArrayParam(i_node, param_name_str);
  } END_ERR_PROP return ret; }
  

  int NeuralGPU_SetNeuronScalVar(int i_node, int n_neurons, char *var_name,
				   float val)
  { int ret; BEGIN_ERR_PROP {
    
    std::string var_name_str = std::string(var_name);
    ret = NeuralGPU_instance->SetNeuronVar(i_node, n_neurons,
					     var_name_str, val);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronArrayVar(int i_node, int n_neurons,
				    char *var_name, float *vars,
				    int array_size)
  { int ret; BEGIN_ERR_PROP {
      std::string var_name_str = std::string(var_name);    
      ret = NeuralGPU_instance->SetNeuronVar(i_node, n_neurons,
					       var_name_str, vars,
					       array_size);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronPtScalVar(int *i_node, int n_neurons,
				     char *var_name,float val)
  { int ret; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    ret = NeuralGPU_instance->SetNeuronVar(i_node, n_neurons,
					     var_name_str, val);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetNeuronPtArrayVar(int *i_node, int n_neurons,
				     char *var_name, float *vars,
				     int array_size)
  { int ret; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);    
    ret = NeuralGPU_instance->SetNeuronVar(i_node, n_neurons,
					     var_name_str, vars,
					     array_size);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronScalVar(int i_node, char *var_name)
  { int ret; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NeuralGPU_instance->IsNeuronScalVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronPortVar(int i_node, char *var_name)
  { int ret; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NeuralGPU_instance->IsNeuronPortVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_IsNeuronArrayVar(int i_node, char *var_name)
  { int ret; BEGIN_ERR_PROP {
    std::string var_name_str = std::string(var_name);
    
    ret = NeuralGPU_instance->IsNeuronArrayVar(i_node, var_name_str);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_Calibrate()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->Calibrate();
  } END_ERR_PROP return ret; }

  int NeuralGPU_Simulate()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->Simulate();
  } END_ERR_PROP return ret; }

  int NeuralGPU_ConnectMpiInit(int argc, char *argv[])
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->ConnectMpiInit(argc, argv);
  } END_ERR_PROP return ret; }

  int NeuralGPU_MpiId()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->MpiId();
  } END_ERR_PROP return ret; }

  int NeuralGPU_MpiNp()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->MpiNp();
  } END_ERR_PROP return ret; }
  int NeuralGPU_ProcMaster()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->ProcMaster();
  } END_ERR_PROP return ret; }

  int NeuralGPU_MpiFinalize()
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->MpiFinalize();
  } END_ERR_PROP return ret; }

  unsigned int *NeuralGPU_RandomInt(size_t n)
  { unsigned int *ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RandomInt(n);
  } END_ERR_PROP return ret; }
  
  float *NeuralGPU_RandomUniform(size_t n)
  { float* ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RandomUniform(n);
  } END_ERR_PROP return ret; }
  
  float *NeuralGPU_RandomNormal(size_t n, float mean, float stddev)
  { float *ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RandomNormal(n, mean, stddev);
  } END_ERR_PROP return ret; }
  
  float *NeuralGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax)
  { float *ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RandomNormalClipped(n, mean, stddev, vmin,
							 vmax);
  } END_ERR_PROP return ret; }
  
  int NeuralGPU_Connect(int i_source_node, int i_target_node,
			unsigned char i_port, float weight, float delay)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->Connect(i_source_node, i_target_node,
				       i_port, weight, delay);
  } END_ERR_PROP return ret; }

  int NeuralGPU_ConnSpecInit()
  { int ret; BEGIN_ERR_PROP {
    ret = ConnSpec_instance.Init();
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetConnSpecParam(char *param_name, int value)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NeuralGPU_ConnSpecIsParam(char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = ConnSpec::IsParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SynSpecInit()
  { int ret; BEGIN_ERR_PROP {
    ret = SynSpec_instance.Init();
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetSynSpecIntParam(char *param_name, int value)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetSynSpecFloatParam(char *param_name, float value)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, value);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.SetParam(param_name_str, array_pt);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SynSpecIsIntParam(char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsIntParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SynSpecIsFloatParam(char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NeuralGPU_SynSpecIsFloatPtParam(char *param_name)
  { int ret; BEGIN_ERR_PROP {
    std::string param_name_str = std::string(param_name);
    ret = SynSpec_instance.IsFloatPtParam(param_name_str);
  } END_ERR_PROP return ret; }

  int NeuralGPU_ConnectSeq(int i_source, int n_source, int i_target,
			   int n_target)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->Connect(
				      i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NeuralGPU_ConnectGroup(int *i_source, int n_source, int *i_target,
			   int n_target)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->Connect(
				      i_source, n_source, i_target, n_target,
				      ConnSpec_instance, SynSpec_instance);
  } END_ERR_PROP return ret; }

  int NeuralGPU_RemoteConnectSeq(int i_source_host, int i_source, int n_source,
				 int i_target_host, int i_target, int n_target)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RemoteConnect(
					     i_source_host, i_source, n_source,
					     i_target_host, i_target, n_target,
					     ConnSpec_instance,
					     SynSpec_instance); 
  } END_ERR_PROP return ret; }

  int NeuralGPU_RemoteConnectGroup(int i_source_host, int *i_source,
				   int n_source,
				   int i_target_host, int *i_target,
				   int n_target)
  { int ret; BEGIN_ERR_PROP {
    ret = NeuralGPU_instance->RemoteConnect(
					     i_source_host, i_source, n_source,
					     i_target_host, i_target, n_target,
					     ConnSpec_instance,
					     SynSpec_instance);
  } END_ERR_PROP return ret; }

}

