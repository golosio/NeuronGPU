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

class BaseNeuron
{
 public:
  int i_node_0_;
  int n_neurons_;
  int n_receptors_;
  int i_neuron_group_;
  int n_var_;
  int n_params_;
  int n_scal_var_;
  int n_vect_var_;
  int n_scal_params_;
  int n_vect_params_;

  float *receptor_weight_arr_;
  int receptor_weight_arr_step_;
  int receptor_weight_port_step_;
  float *receptor_input_arr_;
  int receptor_input_arr_step_;
  int receptor_input_port_step_;
  float *var_arr_;
  float *params_arr_;
  const std::string *scal_var_name_;
  const std::string *vect_var_name_;
  const std::string *scal_param_name_;
  const std::string *vect_param_name_;

  virtual ~BaseNeuron() {}
  
  virtual int Init(int i_node_0, int n_neurons, int n_receptors,
		   int i_neuron_group);

  virtual int Calibrate(float t_min) {return 0;}
		
  virtual int Update(int it, float t1) {return 0;}
  
  virtual int GetX(int i_neuron, int n_neurons, float *x) {return 0;}
  
  virtual int GetY(int i_var, int i_neuron, int n_neurons, float *y) {return 0;}
  
  virtual int SetScalParams(std::string param_name, int i_neuron, int n_neurons,
			    float val);
  
  virtual int SetVectParams(std::string param_name, int i_neuron, int n_neurons,
			    float *params, int vect_size);
  
  virtual int GetScalVarIdx(std::string var_name);

  virtual int GetVectVarIdx(std::string var_name);

  virtual int GetScalParamIdx(std::string param_name);

  virtual int GetVectParamIdx(std::string param_name);

  virtual float *GetVarArr();

  virtual float *GetParamsArr();

  virtual bool IsScalVar(std::string var_name);

  virtual bool IsVectVar(std::string var_name);

  virtual bool IsScalParam(std::string param_name);

  virtual bool IsVectParam(std::string param_name);

  int CheckNeuronIdx(int i_neuron);

  int CheckReceptorIdx(int i_receptor);

  virtual float *GetVarPt(std::string var_name, int i_neuron, int i_receptor);

  virtual float *GetParamPt(std::string param_name, int i_neuron,
		    int i_receptor);
  virtual float GetSpikeActivity(int i_neuron);

};

#endif
