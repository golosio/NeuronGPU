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

#ifndef BASENEURONH
#define BASENEURONH

#include <string>

class BaseNeuron
{
 public:
  int i_node_0_;
  int n_neurons_;
  int n_receptors_;
  int n_var_;
  int n_params_;
  int i_neuron_group_;
  float *G0_;
  
  virtual ~BaseNeuron() {}
  
  virtual int Init(int i_node_0, int n_neurons, int n_receptors,
		   int i_neuron_group, float *G0) {return 0;}

  virtual int Calibrate(float t_min) {return 0;}
		
  virtual int Update(int it, float t1) {return 0;}
  
  virtual int GetX(int i_neuron, int n_neurons, float *x) {return 0;}
  
  virtual int GetY(int i_var, int i_neuron, int n_neurons, float *y) {return 0;}
  
  virtual int SetScalParams(std::string param_name, int i_neuron, int n_neurons,
			float val) {return 0;}
  
  virtual int SetVectParams(std::string param_name, int i_neuron, int n_neurons,
			    float *params, int vect_size) {return 0;}
  
  virtual int GetScalVarIdx(std::string var_name) {return 0;}

  virtual int GetVectVarIdx(std::string var_name) {return 0;}

  virtual float *GetVarArr() {return NULL;}

  virtual float *GetParamsArr() {return NULL;}
};

#endif
