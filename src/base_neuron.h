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

#ifndef BASENEURONH
#define BASENEURONH

#include <string>
#include <vector>
#include "dir_connect.h"
#include <stdint.h>

class NeuronGPU;

class BaseNeuron
{
 protected:
  friend class NeuronGPU;
  int node_type_;
  int i_node_0_;
  int n_node_;
  int n_port_;
  int i_group_;
  unsigned long long *seed_;  

  int n_int_var_;
  int n_scal_var_;
  int n_port_var_;
  int n_scal_param_;
  int n_port_param_;
  int n_var_;
  int n_param_;
  int n_array_var_;
  int n_array_param_;
  
  double *get_spike_array_;
  float *port_weight_arr_;
  int port_weight_arr_step_;
  int port_weight_port_step_;
  float *port_input_arr_;
  int port_input_arr_step_;
  int port_input_port_step_;
  std::vector<int*> int_var_pt_; 
  float *var_arr_;
  float *param_arr_;
  std::vector<std::string> int_var_name_;
  const std::string *scal_var_name_;
  const std::string *port_var_name_;
  const std::string *scal_param_name_;
  const std::string *port_param_name_;
  const std::string *array_var_name_;
  const std::string *array_param_name_;
  
  DirectConnection *d_dir_conn_array_;
  uint64_t n_dir_conn_; // = 0;
  bool has_dir_conn_; // = false;

  int *spike_count_;
  float **rec_spike_times_;
  int *n_rec_spike_times_;
  int max_n_rec_spike_times_;
  
 public:
  virtual ~BaseNeuron() {}
  
  virtual int Init(int i_node_0, int n_neuron, int n_port,
		   int i_neuron_group, unsigned long long *seed);

  virtual int Calibrate(float time_min, float time_resolution) {return 0;}
		
  virtual int Update(int it, float t1) {return 0;}
  
  virtual int GetX(int i_neuron, int n_neuron, float *x) {return 0;}
  
  virtual int GetY(int i_var, int i_neuron, int n_neuron, float *y) {return 0;}
  
  virtual int SetScalParam(int i_neuron, int n_neuron, std::string param_name, 
			   float val);

  virtual int SetScalParam(int *i_neuron, int n_neuron, std::string param_name,
			   float val);
  
  virtual int SetPortParam(int i_neuron, int n_neuron, std::string param_name,
			   float *param, int vect_size);
  
  virtual int SetPortParam(int *i_neuron, int n_neuron,
			   std::string param_name, float *param,
			   int vect_size);

  virtual int SetArrayParam(int i_neuron, int n_neuron, std::string param_name,
			   float *array, int array_size);
  
  virtual int SetArrayParam(int *i_neuron, int n_neuron,
			   std::string param_name, float *array,
			   int array_size);

  virtual int SetIntVar(int i_neuron, int n_neuron, std::string var_name, 
			int val);

  virtual int SetIntVar(int *i_neuron, int n_neuron, std::string var_name,
			int val);

  virtual int SetScalVar(int i_neuron, int n_neuron, std::string var_name, 
			   float val);

  virtual int SetScalVar(int *i_neuron, int n_neuron, std::string var_name,
			   float val);
  
  virtual int SetPortVar(int i_neuron, int n_neuron, std::string var_name,
			   float *var, int vect_size);
  
  virtual int SetPortVar(int *i_neuron, int n_neuron,
			   std::string var_name, float *var,
			   int vect_size);

  virtual int SetArrayVar(int i_neuron, int n_neuron, std::string var_name,
			   float *array, int array_size);
  
  virtual int SetArrayVar(int *i_neuron, int n_neuron,
			   std::string var_name, float *array,
			   int array_size);

  virtual float *GetScalParam(int i_neuron, int n_neuron,
			      std::string param_name);

  virtual float *GetScalParam(int *i_neuron, int n_neuron,
		    std::string param_name);

  virtual float *GetPortParam(int i_neuron, int n_neuron,
			      std::string param_name);

  virtual float *GetPortParam(int *i_neuron, int n_neuron,
			      std::string param_name);

  virtual float *GetArrayParam(int i_neuron, std::string param_name);

  virtual int *GetIntVar(int i_neuron, int n_neuron,
			   std::string var_name);

  virtual int *GetIntVar(int *i_neuron, int n_neuron,
			 std::string var_name);

  virtual float *GetScalVar(int i_neuron, int n_neuron,
			    std::string var_name);

  virtual float *GetScalVar(int *i_neuron, int n_neuron,
			    std::string var_name);

  virtual float *GetPortVar(int i_neuron, int n_neuron,
			    std::string var_name);

  virtual float *GetPortVar(int *i_neuron, int n_neuron,
			    std::string var_name);

  virtual float *GetArrayVar(int i_neuron, std::string var_name);

  virtual int GetIntVarIdx(std::string var_name);
  
  virtual int GetScalVarIdx(std::string var_name);

  virtual int GetPortVarIdx(std::string var_name);

  virtual int GetScalParamIdx(std::string param_name);

  virtual int GetPortParamIdx(std::string param_name);

  virtual float *GetVarArr();

  virtual float *GetParamArr();

  virtual int GetArrayVarSize(int i_neuron, std::string var_name);
  
  virtual int GetArrayParamSize(int i_neuron, std::string param_name);

  virtual int GetVarSize(std::string var_name);

  virtual int GetParamSize(std::string param_name);

  virtual bool IsIntVar(std::string var_name);

  virtual bool IsScalVar(std::string var_name);

  virtual bool IsPortVar(std::string var_name);

  virtual bool IsArrayVar(std::string var_name);
  
  virtual bool IsScalParam(std::string param_name);

  virtual bool IsPortParam(std::string param_name);

  virtual bool IsArrayParam(std::string param_name);

  int CheckNeuronIdx(int i_neuron);

  int CheckPortIdx(int port);

  virtual int *GetIntVarPt(int i_neuron, std::string var_name);
  
  virtual float *GetVarPt(int i_neuron, std::string var_name, int port=0);

  virtual float *GetParamPt(int i_neuron, std::string param_name, 
			    int port=0);
  virtual float GetSpikeActivity(int i_neuron);

  virtual int SendDirectSpikes(float t, float time_step) {return 0;}

  virtual std::vector<std::string> GetIntVarNames();

  virtual int GetNIntVar();
  
  virtual std::vector<std::string> GetScalVarNames();
  
  virtual int GetNScalVar();

  virtual std::vector<std::string> GetPortVarNames();
  
  virtual int GetNPortVar();

  virtual std::vector<std::string> GetScalParamNames();
  
  virtual int GetNScalParam();

  virtual std::vector<std::string> GetPortParamNames();
  
  virtual int GetNPortParam();

  virtual std::vector<std::string> GetArrayVarNames();
  
  virtual int GetNArrayVar();

  virtual std::vector<std::string> GetArrayParamNames();
  
  virtual int GetNArrayParam();

  virtual int ActivateSpikeCount();


};

#endif
