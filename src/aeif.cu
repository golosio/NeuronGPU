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

#include <math.h>
#include <iostream>
#include "rk5.h"
#include "aeif.h"
#include "aeif_variables.h"

__device__
void VarInit(int array_size, int n_var, int n_params, float x, float *y,
	     float *params)
{
  int n_receptors = (n_var-N_SCAL_VAR)/N_VECT_VAR;

  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    V_th = -50.4;
    Delta_T = 2.0;
    g_L = 30.0;
    E_L = -70.6;
    C_m = 281.0;
    a = 4.0;
    b = 80.5;
    tau_w = 144.0;
    I_e = 0.0;
    V_peak = 0.0;
    V_reset = -60.0;
    n_refractory_steps = 1;

    V_m = E_L;
    w = 0;
    refractory_step = 0;
    for (int i = 0; i<n_receptors; i++) {
      g(i) = 0;
      g1(i) = 0;
      E_rev(i) = 0.0;
      taus_decay(i) = 20.0;
      taus_rise(i) = 2.0;
    }
  }
}

__device__
void VarCalibrate(int array_size, int n_var, int n_params, float x, float *y,
		  float *params)
{
  int n_receptors = (n_var-N_SCAL_VAR)/N_VECT_VAR;

  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    V_m = E_L;
    w = 0;
    refractory_step = 0;
    for (int i = 0; i<n_receptors; i++) {
      g(i) = 0;
      g1(i) = 0;
      // denominator is computed here to check that it is != 0
      float denom1 = taus_decay(i) - taus_rise(i);
      float denom2 = 0;
      if (denom1 != 0) {
	// peak time
	float t_p = taus_decay(i)*taus_rise(i)
	  *log(taus_decay(i)/taus_rise(i)) / denom1;
	// another denominator is computed here to check that it is != 0
	denom2 = exp(-t_p / taus_decay(i))
	  - exp(-t_p / taus_rise(i));
      }
      if (denom2 == 0) { // if rise time == decay time use alpha function
	// use normalization for alpha function in this case
	g0(i) = M_E / taus_decay(i);
      }
      else { // if rise time != decay time use beta function
	g0(i) // normalization factor for conductance
	  = ( 1. / taus_rise(i) - 1. / taus_decay(i) ) / denom2;
      }
    }
  }
}


template <>
int AEIF::UpdateNR<0>(int it, float t1)
{
  return 0;
}

int AEIF::Init(int i_node_0, int n_neurons, int n_receptors,
	       int i_neuron_group) {
  h_min_=1.0e-4;
  h_ = 1.0e-2;
  i_node_0_ = i_node_0;
  n_neurons_ = n_neurons;
  n_receptors_ = n_receptors;
  n_var_ = N_SCAL_VAR + N_VECT_VAR*n_receptors;
  n_params_ = N_SCAL_PARAMS + N_VECT_PARAMS*n_receptors;
  i_neuron_group_ = i_neuron_group;
  
  rk5_.Init(n_neurons_, n_var_, n_params_, 0.0, h_);

  receptor_weight_arr_ = GetParamsArr() + N_SCAL_PARAMS
    + GetVectParamIdx("g0");
  receptor_weight_arr_step_ = n_params_;
  receptor_weight_port_step_ = N_VECT_PARAMS;

  receptor_input_arr_ = GetVarArr() + N_SCAL_VAR
    + GetVectVarIdx("g1");
  receptor_input_arr_step_ = n_var_;
  receptor_input_port_step_ = N_VECT_VAR;

  return 0;
}

int AEIF::Calibrate(float t_min) {
  rk5_.Calibrate(t_min, h_);
  
  return 0;
}

int AEIF::Update(int it, float t1) {
  UpdateNR<MAX_RECEPTOR_NUM>(it, t1);

  return 0;
}

int AEIF::SetScalParams(std::string param_name, int i_neuron,
		    int n_neurons, float val) {

  int i_param;
  for (i_param=0; i_param<N_SCAL_PARAMS; i_param++) {
    if (param_name == aeif_scal_param_names[i_param]) break;
  }
  if (i_param == N_SCAL_PARAMS) {
    std::cerr << "Unrecognized parameter " << param_name << " .\n";
    exit(-1);
  }
  	 
  return rk5_.SetParams(i_param, i_neuron, n_params_, n_neurons, val);
}

int AEIF::SetVectParams(std::string param_name, int i_neuron, int n_neurons,
			float *params, int vect_size) {

  int i_vect;
  for (i_vect=0; i_vect<N_VECT_PARAMS; i_vect++) {
    if (param_name == aeif_vect_param_names[i_vect]) break;
  }
  if (i_vect == N_VECT_PARAMS) {
    std::cerr << "Unrecognized vector parameter " << param_name << " \n";
    exit(-1);
  }  
  if (vect_size != n_receptors_) {
    std::cerr << "Parameter vector size must be equal to the number "
      "of receptor ports.\n";
    exit(-1);
  }  

  for (int i=0; i<vect_size; i++) {
    int i_param = N_SCAL_PARAMS + N_VECT_PARAMS*i + i_vect;
    rk5_.SetParams(i_param, i_neuron, n_params_, n_neurons, params[i]);
  }
  return 0;
}

int AEIF::GetScalVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<N_SCAL_VAR; i_var++) {
    if (var_name == aeif_scal_var_names[i_var]) break;
  }
  if (i_var == N_SCAL_VAR) {
    std::cerr << "Unrecognized variable " << var_name << " .\n";
    exit(-1);
  }
  
  return i_var;
}

int AEIF::GetVectVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<N_VECT_VAR; i_var++) {
    if (var_name == aeif_vect_var_names[i_var]) break;
  }
  if (i_var == N_VECT_VAR) {
    std::cerr << "Unrecognized variable " << var_name << " .\n";
    exit(-1);
  }
  
  return i_var;
}

int AEIF::GetScalParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<N_SCAL_PARAMS; i_param++) {
    if (param_name == aeif_scal_param_names[i_param]) break;
  }
  if (i_param == N_SCAL_PARAMS) {
    std::cerr << "Unrecognized parameter " << param_name << " .\n";
    exit(-1);
  }
  
  return i_param;
}

int AEIF::GetVectParamIdx(std::string param_name)
{
  if (param_name=="receptor_weight") return GetVectParamIdx("g0");
  
  int i_param;
  for (i_param=0; i_param<N_VECT_PARAMS; i_param++) {
    if (param_name == aeif_vect_param_names[i_param]) break;
  }
  if (i_param == N_VECT_PARAMS) {
    std::cerr << "Unrecognized vector parameter " << param_name << " .\n";
    exit(-1);
  }
  
  return i_param;
}

float *AEIF::GetVarArr()
{
  return rk5_.GetYArr();
}

float *AEIF::GetParamsArr()
{
  return rk5_.GetParamsArr();
}
