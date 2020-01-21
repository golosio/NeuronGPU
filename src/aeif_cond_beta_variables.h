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

#ifndef AEIFCONDBETAVARIABLESH
#define AEIFCONDBETAVARIABLESH

#include <string>
#include <math.h>
#include "spike_buffer.h"
#include "node_group.h"
#include "aeif_cond_beta.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

enum ScalVarIndexes {
  i_V_m = 0,
  i_w,
  N_SCAL_VAR
};

enum PortVarIndexes {
  i_g = 0,
  i_g1,
  N_PORT_VAR
};

enum ScalParamIndexes {
  i_V_th = 0,
  i_Delta_T,
  i_g_L,
  i_E_L,
  i_C_m,
  i_a,
  i_b,
  i_tau_w,
  i_I_e,
  i_V_peak,
  i_V_reset,
  i_n_refractory_steps,
  i_refractory_step,
  N_SCAL_PARAMS
};

enum PortParamIndexes {
  i_E_rev = 0,
  i_taus_rise,
  i_taus_decay,
  i_g0,
  N_PORT_PARAMS
};

const std::string aeif_cond_beta_scal_var_name[N_SCAL_VAR] = {
  "V_m",
  "w"
};

const std::string aeif_cond_beta_port_var_name[N_PORT_VAR] = {
  "g",
  "g1"
};

const std::string aeif_cond_beta_scal_param_name[N_SCAL_PARAMS] = {
  "V_th",
  "Delta_T",
  "g_L",
  "E_L",
  "C_m",
  "a",
  "b",
  "tau_w",
  "I_e",
  "V_peak",
  "V_reset",
  "n_refractory_steps",
  "refractory_step"
};

const std::string aeif_cond_beta_port_param_name[N_PORT_PARAMS] = {
  "E_rev",
  "taus_rise",
  "taus_decay",
  "g0"  
};

//
// I know that defines are "bad", but the defines below make the
// following equations much more readable.
// For every rule there is some exceptions!
//
#define V_m y[i_V_m]
#define w y[i_w]
#define g(i) y[N_SCAL_VAR + N_PORT_VAR*i + i_g]
#define g1(i) y[N_SCAL_VAR + N_PORT_VAR*i + i_g1]

#define dVdt dydx[i_V_m]
#define dwdt dydx[i_w]
#define dgdt(i) dydx[N_SCAL_VAR + N_PORT_VAR*i + i_g]
#define dg1dt(i) dydx[N_SCAL_VAR + N_PORT_VAR*i + i_g1]

#define V_th params[i_V_th]
#define Delta_T params[i_Delta_T]
#define g_L params[i_g_L]
#define E_L params[i_E_L]
#define C_m params[i_C_m]
#define a params[i_a]
#define b params[i_b]
#define tau_w params[i_tau_w]
#define I_e params[i_I_e]
#define V_peak params[i_V_peak]
#define V_reset params[i_V_reset]
#define n_refractory_steps params[i_n_refractory_steps]
#define refractory_step params[i_refractory_step]

#define E_rev(i) params[N_SCAL_PARAMS + N_PORT_PARAMS*i + i_E_rev]
#define taus_rise(i) params[N_SCAL_PARAMS + N_PORT_PARAMS*i + i_taus_rise]
#define taus_decay(i) params[N_SCAL_PARAMS + N_PORT_PARAMS*i + i_taus_decay]
#define g0(i) params[N_SCAL_PARAMS + N_PORT_PARAMS*i + i_g0]


template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void aeif_cond_beta_Derivatives(float x, float *y, float *dydx, float *params,
		     DataStruct data_struct)
{
  enum { n_ports = (NVAR-N_SCAL_VAR)/N_PORT_VAR };
  float I_syn = 0.0;

  float V = ( refractory_step > 0 ) ? V_reset :  MIN(V_m, V_peak);
  for (int i = 0; i<n_ports; i++) {
    I_syn += g(i)*(E_rev(i) - V);
  }
  float V_spike = Delta_T*exp((V - V_th)/Delta_T);

  dVdt = ( refractory_step > 0 ) ? 0 :
    ( -g_L*(V - E_L - V_spike) + I_syn - w + I_e) / C_m;
  // Adaptation current w.
  dwdt = (a*(V - E_L) - w) / tau_w;
  for (int i=0; i<n_ports; i++) {
    // Synaptic conductance derivative
    dg1dt(i) = -g1(i) / taus_rise(i);
    dgdt(i) = g1(i) - g(i) / taus_decay(i);
  }
}

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void aeif_cond_beta_ExternalUpdate
    (float x, float *y, float *params, bool end_time_step,
			RK5DataStruct data_struct)
{
  if ( V_m < -1.0e3) { // numerical instability
    printf("V_m out of lower bound\n");
    V_m = V_reset;
    w=0;
    return;
  }
  if ( w < -1.0e6 || w > 1.0e6) { // numerical instability
    printf("w out of bound\n");
    V_m = V_reset;
    w=0;
    return;
  }
  if (refractory_step > 0) {
    V_m = V_reset;
    if (end_time_step) {
      refractory_step--;
    }
  }
  else {
    if ( V_m >= V_peak ) { // send spike
      int neuron_idx = threadIdx.x + blockIdx.x * blockDim.x;
      PushSpike(data_struct.i_node_0_ + neuron_idx, 1.0);
      V_m = V_reset;
      w += b; // spike-driven adaptation
      refractory_step = n_refractory_steps;
    }
  }
}

template<class DataStruct>
__device__
void aeif_cond_beta_NodeInit(int n_var, int n_params, float x, float *y, float *params,
		  DataStruct data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n_ports = (n_var-N_SCAL_VAR)/N_PORT_VAR;

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
  for (int i = 0; i<n_ports; i++) {
    g(i) = 0;
    g1(i) = 0;
    E_rev(i) = 0.0;
    taus_decay(i) = 20.0;
    taus_rise(i) = 2.0;
  }
}

template<class DataStruct>
__device__
void aeif_cond_beta_NodeCalibrate(int n_var, int n_params, float x, float *y,
		       float *params, DataStruct data_struct)
{
  //int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int n_ports = (n_var-N_SCAL_VAR)/N_PORT_VAR;

  refractory_step = 0;
  for (int i = 0; i<n_ports; i++) {
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

template <>
int aeif_cond_beta::UpdateNR<0>(int it, float t1);

template<int N_PORTS>
int aeif_cond_beta::UpdateNR(int it, float t1)
{
  if (N_PORTS == n_ports_) {
    const int NVAR = N_SCAL_VAR + N_PORT_VAR*N_PORTS;
    const int NPARAMS = N_SCAL_PARAMS + N_PORT_PARAMS*N_PORTS;

    rk5_.Update<NVAR, NPARAMS>(t1, h_min_, rk5_data_struct_);
  }
  else {
    UpdateNR<N_PORTS - 1>(it, t1);
  }

  return 0;
}

#endif
