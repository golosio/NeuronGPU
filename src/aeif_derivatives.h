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

#ifndef AEIFDERIVATIVESH
#define AEIFDERIVATIVESH

#include <string>
#include <stdio.h>
#include <math.h>
#include "spike_buffer.h"
#include "neuron_group.h"
#include "aeif_variables.h"

#define MIN(a,b) (((a)<(b))?(a):(b))

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void AEIF_Derivatives(float x, float *y, float *dydx, float *params,
		     DataStruct data_struct)
{
  enum { n_receptors = (NVAR-N_SCAL_VAR)/N_VECT_VAR };
  float I_syn = 0.0;

  float V = ( refractory_step > 0 ) ? V_reset :  MIN(V_m, V_peak);
  for (int i = 0; i<n_receptors; i++) {
    I_syn += g(i)*(E_rev(i) - V);
  }
  float V_spike = Delta_T*exp((V - V_th)/Delta_T);

  dVdt = ( refractory_step > 0 ) ? 0 :
    ( -g_L*(V - E_L - V_spike) + I_syn - w + I_e) / C_m;
  // Adaptation current w.
  dwdt = (a*(V - E_L) - w) / tau_w;
  for (int i=0; i<n_receptors; i++) {
    // Synaptic conductance derivative
    dg1dt(i) = -g1(i) / taus_rise(i);
    dgdt(i) = g1(i) - g(i) / taus_decay(i);
  }
}

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void AEIF_ExternalUpdate
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
      PushSpike(data_struct.i_neuron_0_ + neuron_idx, 1.0);
      V_m = V_reset;
      w += b; // spike-driven adaptation
      refractory_step = n_refractory_steps;
    }
  }
}

#endif
