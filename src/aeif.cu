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
  n_scal_var_ = N_SCAL_VAR;
  n_vect_var_ = N_VECT_VAR;
  n_scal_params_ = N_SCAL_PARAMS;
  n_vect_params_ = N_VECT_PARAMS;

  n_var_ = n_scal_var_ + n_vect_var_*n_receptors;
  n_params_ = n_scal_params_ + n_vect_params_*n_receptors;
  i_neuron_group_ = i_neuron_group;
  scal_var_name_ = aeif_scal_var_name;
  vect_var_name_= aeif_vect_var_name;
  scal_param_name_ = aeif_scal_param_name;
  vect_param_name_ = aeif_vect_param_name;

  rk5_.Init(n_neurons_, n_var_, n_params_, 0.0, h_);
  var_arr_ = rk5_.GetYArr();
  params_arr_ = rk5_.GetParamsArr();

  receptor_weight_arr_ = GetParamsArr() + n_scal_params_
    + GetVectParamIdx("g0");
  receptor_weight_arr_step_ = n_params_;
  receptor_weight_port_step_ = n_vect_params_;

  receptor_input_arr_ = GetVarArr() + n_scal_var_
    + GetVectVarIdx("g1");
  receptor_input_arr_step_ = n_var_;
  receptor_input_port_step_ = n_vect_var_;


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

