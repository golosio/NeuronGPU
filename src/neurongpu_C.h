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

#ifndef NEURONGPUH
#define NEURONGPUH

#ifdef __cplusplus
extern "C" {
#endif
  
  char *NeuronGPU_GetErrorMessage();

  unsigned char NeuronGPU_GetErrorCode();
  
  void NeuronGPU_SetOnException(int on_exception);

  int NeuronGPU_SetRandomSeed(unsigned long long seed);

  int NeuronGPU_SetTimeResolution(float time_res);

  float NeuronGPU_GetTimeResolution();

  int NeuronGPU_SetMaxSpikeBufferSize(int max_size);

  int NeuronGPU_GetMaxSpikeBufferSize();

  int NeuronGPU_Create(char *model_name, int n_neuron, int n_port);

  int NeuronGPU_CreatePoissonGenerator(int n_node, float rate);
  
  int NeuronGPU_CreateRecord(char *file_name, char *var_name_arr[],
			     int *i_node_arr, int *port_arr,
			     int n_node);
  
  int NeuronGPU_GetRecordDataRows(int i_record);
  
  int NeuronGPU_GetRecordDataColumns(int i_record);

  float **NeuronGPU_GetRecordData(int i_record);

  int NeuronGPU_SetNeuronScalParam(int i_node, int n_neuron, char *param_name,
				   float val);

  int NeuronGPU_SetNeuronArrayParam(int i_node, int n_neuron,
				    char *param_name, float *param,
				    int array_size);

  int NeuronGPU_SetNeuronPtScalParam(int *i_node, int n_neuron,
				     char *param_name, float val);

  int NeuronGPU_SetNeuronPtArrayParam(int *i_node, int n_neuron,
				      char *param_name, float *param,
				      int array_size);
  
  int NeuronGPU_IsNeuronScalParam(int i_node, char *param_name);
  
  int NeuronGPU_IsNeuronPortParam(int i_node, char *param_name);

  int NeuronGPU_IsNeuronArrayParam(int i_node, char *param_name);
  

  int NeuronGPU_SetNeuronIntVar(int i_node, int n_neuron, char *var_name,
				int val);

  int NeuronGPU_SetNeuronScalVar(int i_node, int n_neuron, char *var_name,
				   float val);

  int NeuronGPU_SetNeuronArrayVar(int i_node, int n_neuron,
				    char *var_name, float *var,
				    int array_size);

  int NeuronGPU_SetNeuronPtIntVar(int *i_node, int n_neuron,
				  char *var_name, int val);

  int NeuronGPU_SetNeuronPtScalVar(int *i_node, int n_neuron,
				     char *var_name, float val);

  int NeuronGPU_SetNeuronPtArrayVar(int *i_node, int n_neuron,
				      char *var_name, float *var,
				      int array_size);
  
  int NeuronGPU_IsNeuronIntVar(int i_node, char *var_name);

  int NeuronGPU_IsNeuronScalVar(int i_node, char *var_name);
  
  int NeuronGPU_IsNeuronPortVar(int i_node, char *var_name);

  int NeuronGPU_IsNeuronArrayVar(int i_node, char *var_name);

  int NeuronGPU_GetNeuronParamSize(int i_node, char *param_name);

  int NeuronGPU_GetNeuronVarSize(int i_node, char *var_name);
  
  float *NeuronGPU_GetNeuronParam(int i_node, int n_neuron,
				  char *param_name);

  float *NeuronGPU_GetNeuronPtParam(int *i_node, int n_neuron,
				    char *param_name);

  float *NeuronGPU_GetArrayParam(int i_node, char *param_name);

  int *NeuronGPU_GetNeuronIntVar(int i_node, int n_neuron,
				 char *param_name);

  int *NeuronGPU_GetNeuronPtIntVar(int *i_node, int n_neuron,
				   char *param_name);
  
  float *NeuronGPU_GetNeuronVar(int i_node, int n_neuron,
				char *param_name);

  float *NeuronGPU_GetNeuronPtVar(int *i_node, int n_neuron,
				  char *param_name);
  
  float *NeuronGPU_GetArrayVar(int i_node, char *var_name);
  
  int NeuronGPU_Calibrate();

  int NeuronGPU_Simulate();

  int NeuronGPU_ConnectMpiInit(int argc, char *argv[]);

  int NeuronGPU_MpiId();

  int NeuronGPU_MpiNp();

  int NeuronGPU_ProcMaster();

  int NeuronGPU_MpiFinalize();

  unsigned int *NeuronGPU_RandomInt(size_t n);
  
  float *NeuronGPU_RandomUniform(size_t n);
  
  float *NeuronGPU_RandomNormal(size_t n, float mean, float stddev);
  
  float *NeuronGPU_RandomNormalClipped(size_t n, float mean, float stddev,
				       float vmin, float vmax);
  
  int NeuronGPU_Connect(int i_source_node, int i_target_node,
			unsigned char port, unsigned char syn_group,
			float weight, float delay);

  int NeuronGPU_ConnSpecInit();

  int NeuronGPU_SetConnSpecParam(char *param_name, int value);

  int NeuronGPU_ConnSpecIsParam(char *param_name);

  int NeuronGPU_SynSpecInit();

  int NeuronGPU_SetSynSpecIntParam(char *param_name, int value);

  int NeuronGPU_SetSynSpecFloatParam(char *param_name, float value);

  int NeuronGPU_SetSynSpecFloatPtParam(char *param_name, float *array_pt);

  int NeuronGPU_SynSpecIsIntParam(char *param_name);

  int NeuronGPU_SynSpecIsFloatParam(char *param_name);

  int NeuronGPU_SynSpecIsFloatPtParam(char *param_name);

  int NeuronGPU_ConnectSeqSeq(int i_source, int n_source, int i_target,
			      int n_target);

  int NeuronGPU_ConnectSeqGroup(int i_source, int n_source, int *i_target,
				int n_target);

  int NeuronGPU_ConnectGroupSeq(int *i_source, int n_source, int i_target,
				int n_target);

  int NeuronGPU_ConnectGroupGroup(int *i_source, int n_source, int *i_target,
				  int n_target);

  int NeuronGPU_RemoteConnectSeqSeq(int i_source_host, int i_source,
				    int n_source, int i_target_host,
				    int i_target, int n_target);

  int NeuronGPU_RemoteConnectSeqGroup(int i_source_host, int i_source,
				      int n_source, int i_target_host,
				      int *i_target, int n_target);

  int NeuronGPU_RemoteConnectGroupSeq(int i_source_host, int *i_source,
				      int n_source, int i_target_host,
				      int i_target, int n_target);

  int NeuronGPU_RemoteConnectGroupGroup(int i_source_host, int *i_source,
					int n_source, int i_target_host,
					int *i_target, int n_target);

  char **NeuronGPU_GetIntVarNames(int i_node);

  char **NeuronGPU_GetScalVarNames(int i_node);
  
  int NeuronGPU_GetNIntVar(int i_node);

  int NeuronGPU_GetNScalVar(int i_node);
    
  char **NeuronGPU_GetPortVarNames(int i_node);
  
  int NeuronGPU_GetNPortVar(int i_node);
    
  char **NeuronGPU_GetScalParamNames(int i_node);
  
  int NeuronGPU_GetNScalParam(int i_node);
    
  char **NeuronGPU_GetPortParamNames(int i_node);
  
  int NeuronGPU_GetNPortParam(int i_node);

  char **NeuronGPU_GetArrayParamNames(int i_node);
  
  int NeuronGPU_GetNArrayParam(int i_node);

  char **NeuronGPU_GetArrayVarNames(int i_node);
  
  int NeuronGPU_GetNArrayVar(int i_node);
    
  int *NeuronGPU_GetSeqSeqConnections(int i_source, int n_source, int i_target,
				      int n_target, int syn_group, int *n_conn);

  int *NeuronGPU_GetSeqGroupConnections(int i_source, int n_source,
					int *i_target, int n_target,
					int syn_group, int *n_conn);

  int *NeuronGPU_GetGroupSeqConnections(int *i_source, int n_source,
					int i_target, int n_target,
					int syn_group, int *n_conn);

  int *NeuronGPU_GetGroupGroupConnections(int *i_source, int n_source,
					  int *i_target, int n_target,
					  int syn_group, int *n_conn);
  int NeuronGPU_GetConnectionStatus(int i_source, int i_group, int i_conn,
				    int *i_target, unsigned char *port,
				    unsigned char *syn_group, float *delay,
				    float *weight);

  int NeuronGPU_CreateSynGroup(char *model_name);
  
  int NeuronGPU_GetSynGroupNParam(int i_syn_group);
  
  char **NeuronGPU_GetSynGroupParamNames(int i_syn_group);
  
  int NeuronGPU_IsSynGroupParam(int i_syn_group, char *param_name);
  
  int NeuronGPU_GetSynGroupParamIdx(int i_syn_group, char *param_name);
  
  float NeuronGPU_GetSynGroupParam(int i_syn_group, char *param_name);
  
  int NeuronGPU_SetSynGroupParam(int i_syn_group, char *param_name, float val);

  int NeuronGPU_ActivateSpikeCount(int i_node, int n_node);

#ifdef __cplusplus
}
#endif


#endif
