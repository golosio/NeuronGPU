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

  int NeuralGPU_Create(char *model_name, int n_neuron, int n_port);

  int NeuralGPU_CreatePoissonGenerator(int n_node, float rate);
  
  int NeuralGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *port_arr,
			     int n_node);
  
  int NeuralGPU_GetRecordDataRows(int i_record);
  
  int NeuralGPU_GetRecordDataColumns(int i_record);

  float **NeuralGPU_GetRecordData(int i_record);

  int NeuralGPU_SetNeuronScalParam(int i_node, int n_neuron, char *param_name,
				   float val);

  int NeuralGPU_SetNeuronArrayParam(int i_node, int n_neuron,
				    char *param_name, float *param,
				    int array_size);

  int NeuralGPU_SetNeuronPtScalParam(int *i_node, int n_neuron,
				     char *param_name, float val);

  int NeuralGPU_SetNeuronPtArrayParam(int *i_node, int n_neuron,
				      char *param_name, float *param,
				      int array_size);
  
  int NeuralGPU_IsNeuronScalParam(int i_node, char *param_name);
  
  int NeuralGPU_IsNeuronPortParam(int i_node, char *param_name);

  int NeuralGPU_IsNeuronArrayParam(int i_node, char *param_name);
  

  int NeuralGPU_SetNeuronScalVar(int i_node, int n_neuron, char *var_name,
				   float val);

  int NeuralGPU_SetNeuronArrayVar(int i_node, int n_neuron,
				    char *var_name, float *var,
				    int array_size);

  int NeuralGPU_SetNeuronPtScalVar(int *i_node, int n_neuron,
				     char *var_name, float val);

  int NeuralGPU_SetNeuronPtArrayVar(int *i_node, int n_neuron,
				      char *var_name, float *var,
				      int array_size);
  
  int NeuralGPU_IsNeuronScalVar(int i_node, char *var_name);
  
  int NeuralGPU_IsNeuronPortVar(int i_node, char *var_name);

  int NeuralGPU_IsNeuronArrayVar(int i_node, char *var_name);

  int NeuralGPU_GetNeuronParamSize(int i_node, char *param_name);

  int NeuralGPU_GetNeuronVarSize(int i_node, char *var_name);
  
  float *NeuralGPU_GetNeuronParam(int i_node, int n_neuron,
				  char *param_name);

  float *NeuralGPU_GetNeuronPtParam(int *i_node, int n_neuron,
				    char *param_name);

  float *NeuralGPU_GetArrayParam(int i_node, char *param_name);

  float *NeuralGPU_GetNeuronVar(int i_node, int n_neuron,
				char *param_name);

  float *NeuralGPU_GetNeuronPtVar(int *i_node, int n_neuron,
				  char *param_name);
  
  float *NeuralGPU_GetArrayVar(int i_node, char *var_name);
  
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
			unsigned char port, unsigned char syn_group,
			float weight, float delay);

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

  int NeuralGPU_ConnectSeqSeq(int i_source, int n_source, int i_target,
			      int n_target);

  int NeuralGPU_ConnectSeqGroup(int i_source, int n_source, int *i_target,
				int n_target);

  int NeuralGPU_ConnectGroupSeq(int *i_source, int n_source, int i_target,
				int n_target);

  int NeuralGPU_ConnectGroupGroup(int *i_source, int n_source, int *i_target,
				  int n_target);

  int NeuralGPU_RemoteConnectSeqSeq(int i_source_host, int i_source,
				    int n_source, int i_target_host,
				    int i_target, int n_target);

  int NeuralGPU_RemoteConnectSeqGroup(int i_source_host, int i_source,
				      int n_source, int i_target_host,
				      int *i_target, int n_target);

  int NeuralGPU_RemoteConnectGroupSeq(int i_source_host, int *i_source,
				      int n_source, int i_target_host,
				      int i_target, int n_target);

  int NeuralGPU_RemoteConnectGroupGroup(int i_source_host, int *i_source,
					int n_source, int i_target_host,
					int *i_target, int n_target);

  char **NeuralGPU_GetScalVarNames(int i_node);
  
  int NeuralGPU_GetNScalVar(int i_node);
    
  char **NeuralGPU_GetPortVarNames(int i_node);
  
  int NeuralGPU_GetNPortVar(int i_node);
    
  char **NeuralGPU_GetScalParamNames(int i_node);
  
  int NeuralGPU_GetNScalParam(int i_node);
    
  char **NeuralGPU_GetPortParamNames(int i_node);
  
  int NeuralGPU_GetNPortParam(int i_node);

  char **NeuralGPU_GetArrayParamNames(int i_node);
  
  int NeuralGPU_GetNArrayParam(int i_node);

  char **NeuralGPU_GetArrayVarNames(int i_node);
  
  int NeuralGPU_GetNArrayVar(int i_node);
    
  int *NeuralGPU_GetSeqSeqConnections(int i_source, int n_source, int i_target,
				      int n_target, int syn_group, int *n_conn);

  int *NeuralGPU_GetSeqGroupConnections(int i_source, int n_source,
					int *i_target, int n_target,
					int syn_group, int *n_conn);

  int *NeuralGPU_GetGroupSeqConnections(int *i_source, int n_source,
					int i_target, int n_target,
					int syn_group, int *n_conn);

  int *NeuralGPU_GetGroupGroupConnections(int *i_source, int n_source,
					  int *i_target, int n_target,
					  int syn_group, int *n_conn);
  int NeuralGPU_GetConnectionStatus(int i_source, int i_group, int i_conn,
				    int *i_target, unsigned char *port,
				    unsigned char *syn_group, float *delay,
				    float *weight);

#ifdef __cplusplus
}
#endif


#endif
