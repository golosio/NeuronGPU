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

#ifndef NEURALGPUH
#define NEURALGPUH

#ifdef __cplusplus
extern "C" {
#endif
  
  char *NeuralGPU_GetErrorMessage();

  unsigned char NeuralGPU_GetErrorCode();
  
  void NeuralGPU_SetOnException(int on_exception);

  int NeuralGPU_SetRandomSeed(unsigned long long seed);

  int NeuralGPU_SetTimeResolution(float time_res);

  float NeuralGPU_GetTimeResolution();

  int NeuralGPU_SetMaxSpikeBufferSize(int max_size);

  int NeuralGPU_GetMaxSpikeBufferSize();

  int NeuralGPU_Create(char *model_name, int n_neurons, int n_ports);

  int NeuralGPU_CreatePoissonGenerator(int n_nodes, float rate);
  
  int NeuralGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *i_port_arr,
			     int n_nodes);
  
  int NeuralGPU_GetRecordDataRows(int i_record);
  
  int NeuralGPU_GetRecordDataColumns(int i_record);

  float **NeuralGPU_GetRecordData(int i_record);

  int NeuralGPU_SetNeuronScalParam(int i_node, int n_neurons, char *param_name,
				   float val);

  int NeuralGPU_SetNeuronArrayParam(int i_node, int n_neurons,
				    char *param_name, float *params,
				    int array_size);

  int NeuralGPU_SetNeuronPtScalParam(int *i_node, int n_neurons,
				     char *param_name, float val);

  int NeuralGPU_SetNeuronPtArrayParam(int *i_node, int n_neurons,
				      char *param_name, float *params,
				      int array_size);
  
  int NeuralGPU_IsNeuronScalParam(int i_node, char *param_name);
  
  int NeuralGPU_IsNeuronPortParam(int i_node, char *param_name);

  int NeuralGPU_IsNeuronArrayParam(int i_node, char *param_name);
  

  int NeuralGPU_SetNeuronScalVar(int i_node, int n_neurons, char *var_name,
				   float val);

  int NeuralGPU_SetNeuronArrayVar(int i_node, int n_neurons,
				    char *var_name, float *vars,
				    int array_size);

  int NeuralGPU_SetNeuronPtScalVar(int *i_node, int n_neurons,
				     char *var_name, float val);

  int NeuralGPU_SetNeuronPtArrayVar(int *i_node, int n_neurons,
				      char *var_name, float *vars,
				      int array_size);
  
  int NeuralGPU_IsNeuronScalVar(int i_node, char *var_name);
  
  int NeuralGPU_IsNeuronPortVar(int i_node, char *var_name);

  int NeuralGPU_IsNeuronArrayVar(int i_node, char *var_name);

  int NeuralGPU_GetNeuronParamSize(int i_node, char *param_name);

  int NeuralGPU_GetNeuronVarSize(int i_node, char *var_name);
  
  float *NeuralGPU_GetNeuronParam(int i_node, int n_neurons,
				  char *param_name);

  float *NeuralGPU_GetNeuronPtParam(int *i_node, int n_neurons,
				    char *param_name);

  float *NeuralGPU_GetNeuronVar(int i_node, int n_neurons,
				char *param_name);

  float *NeuralGPU_GetNeuronPtVar(int *i_node, int n_neurons,
				  char *param_name);
  
  int NeuralGPU_Calibrate();

  int NeuralGPU_Simulate();

  int NeuralGPU_ConnectMpiInit(int argc, char *argv[]);

  int NeuralGPU_MpiId();

  int NeuralGPU_MpiNp();

  int NeuralGPU_ProcMaster();

  int NeuralGPU_MpiFinalize();

  unsigned int *NeuralGPU_RandomInt(size_t n);
  
  float *NeuralGPU_RandomUniform(size_t n);
  
  float *NeuralGPU_RandomNormal(size_t n, float mean, float stddev);
  
  float *NeuralGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax);
  
  int NeuralGPU_Connect(int i_source_node, int i_target_node,
			unsigned char i_port, float weight, float delay);

  int NeuralGPU_ConnSpecInit();

  int NeuralGPU_SetConnSpecParam(char *param_name, int value);

  int NeuralGPU_ConnSpecIsParam(char *param_name);

  int NeuralGPU_SynSpecInit();

  int NeuralGPU_SetSynSpecIntParam(char *param_name, int value);

  int NeuralGPU_SetSynSpecFloatParam(char *param_name, float value);

  int NeuralGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt);

  int NeuralGPU_SynSpecIsIntParam(char *param_name);

  int NeuralGPU_SynSpecIsFloatParam(char *param_name);

  int NeuralGPU_SynSpecIsFloatPtParam(char *param_name);

  int NeuralGPU_ConnectSeq(int i_source, int n_source, int i_target,
			   int n_target);

  int NeuralGPU_ConnectGroup(int *i_source, int n_source, int *i_target,
			     int n_target);

  int NeuralGPU_RemoteConnectSeq(int i_source_host, int i_source, int n_source,
				 int i_target_host, int i_target, int n_target);

  int NeuralGPU_RemoteConnectGroup(int i_source_host, int *i_source,
				   int n_source,
				   int i_target_host, int *i_target,
				   int n_target);


#ifdef __cplusplus
}
#endif


#endif
