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
#include "dir_connect.h"

class NeuralGPU;

class BaseNeuron
{
 protected:
  friend class NeuralGPU;
  int node_type_;
  int i_node_0_;
  int n_nodes_;
  int n_ports_;
  int i_group_;
  unsigned long long *seed_;  

  int n_scal_var_;
  int n_port_var_;
  int n_scal_params_;
  int n_port_params_;
  int n_var_;
  int n_params_;

  double *get_spike_array_;
  float *port_weight_arr_;
  int port_weight_arr_step_;
  int port_weight_port_step_;
  float *port_input_arr_;
  int port_input_arr_step_;
  int port_input_port_step_;
  float *var_arr_;
  float *params_arr_;
  const std::string *scal_var_name_;
  const std::string *port_var_name_;
  const std::string *scal_param_name_;
  const std::string *port_param_name_;

  DirectConnection *d_dir_conn_array_;
  long n_dir_conn_; // = 0;
  bool has_dir_conn_; // = false;

 public:
  virtual ~BaseNeuron() {}
  
  virtual int Init(int i_node_0, int n_neurons, int n_ports,
		   int i_neuron_group, unsigned long long *seed);

  virtual int Calibrate(float time_min, float time_resolution) {return 0;}
		
  virtual int Update(int it, float t1) {return 0;}
  
  virtual int GetX(int i_neuron, int n_neurons, float *x) {return 0;}
  
  virtual int GetY(int i_var, int i_neuron, int n_neurons, float *y) {return 0;}
  
  virtual int SetScalParam(int i_neuron, int n_neurons, std::string param_name, 
			   float val);

  virtual int SetScalParam(int *i_neuron, int n_neurons, std::string param_name,
			   float val);
  
  virtual int SetPortParam(int i_neuron, int n_neurons, std::string param_name,
			   float *params, int vect_size);
  
  virtual int SetPortParam(int *i_neuron, int n_neurons,
			   std::string param_name, float *params,
			   int vect_size);

  virtual int SetArrayParam(int i_neuron, int n_neurons, std::string param_name,
			   float *array, int array_size);
  
  virtual int SetArrayParam(int *i_neuron, int n_neurons,
			   std::string param_name, float *array,
			   int array_size);

  virtual int SetScalVar(int i_neuron, int n_neurons, std::string var_name, 
			   float val);

  virtual int SetScalVar(int *i_neuron, int n_neurons, std::string var_name,
			   float val);
  
  virtual int SetPortVar(int i_neuron, int n_neurons, std::string var_name,
			   float *vars, int vect_size);
  
  virtual int SetPortVar(int *i_neuron, int n_neurons,
			   std::string var_name, float *vars,
			   int vect_size);

  virtual int SetArrayVar(int i_neuron, int n_neurons, std::string var_name,
			   float *array, int array_size);
  
  virtual int SetArrayVar(int *i_neuron, int n_neurons,
			   std::string var_name, float *array,
			   int array_size);

  virtual float *GetScalParam(int i_neuron, int n_neurons,
			      std::string param_name);

  virtual float *GetScalParam(int *i_neuron, int n_neurons,
		    std::string param_name);

  virtual float *GetPortParam(int i_neuron, int n_neurons,
			      std::string param_name);

  virtual float *GetPortParam(int *i_neuron, int n_neurons,
			      std::string param_name);

  virtual float *GetArrayParam(int i_neuron, int n_neurons,
			       std::string param_name);

  virtual float *GetArrayParam(int *i_neuron, int n_neurons,
			       std::string param_name);

  virtual float *GetScalVar(int i_neuron, int n_neurons,
			    std::string var_name);

  virtual float *GetScalVar(int *i_neuron, int n_neurons,
			    std::string var_name);

  virtual float *GetPortVar(int i_neuron, int n_neurons,
			    std::string var_name);

  virtual float *GetPortVar(int *i_neuron, int n_neurons,
			    std::string var_name);

  virtual float *GetArrayVar(int i_neuron, int n_neurons,
			     std::string var_name);

  virtual float *GetArrayVar(int *i_neuron, int n_neurons,
			     std::string var_name);
  
  virtual int GetScalVarIdx(std::string var_name);

  virtual int GetPortVarIdx(std::string var_name);

  virtual int GetScalParamIdx(std::string param_name);

  virtual int GetPortParamIdx(std::string param_name);

  virtual float *GetVarArr();

  virtual float *GetParamArr();

  virtual bool IsScalVar(std::string var_name);

  virtual bool IsPortVar(std::string var_name);

  virtual bool IsArrayVar(std::string var_name);
  
  virtual bool IsScalParam(std::string param_name);

  virtual bool IsPortParam(std::string param_name);

  virtual bool IsArrayParam(std::string param_name);

  int CheckNeuronIdx(int i_neuron);

  int CheckPortIdx(int i_port);

  virtual float *GetVarPt(int i_neuron, std::string var_name, int i_port=0);

  virtual float *GetParamPt(int i_neuron, std::string param_name, 
			    int i_port=0);
  virtual float GetSpikeActivity(int i_neuron);

  virtual int SendDirectSpikes(float t, float time_step) {return 0;}

};

#endif
