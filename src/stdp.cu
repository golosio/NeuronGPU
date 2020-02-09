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

#include <stdio.h>
#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "stdp.h"

using namespace stdp_ns;

__device__ void STDPUpdate(float *weight_pt, float Dt, float *param)
{
  float tau_plus = param[i_tau_plus];
  float tau_minus = param[i_tau_minus];
  float lambda = param[i_lambda];
  float alpha = param[i_alpha];
  float mu_plus = param[i_mu_plus];
  float mu_minus = param[i_mu_minus];
  float Wmax = param[i_Wmax];
  float Kplus = param[i_Kplus];
  float Kminus = param[i_Kminus];

  float w = *weight_pt;
  if (Dt>=0) {
    float fact = Kplus*exp(-Dt/tau_plus);
    //float norm_w = ((w/Wmax) + (lambda*pow(1.0 - (w/Wmax), mu_plus)))*fact);
    //*weight_pt = norm_w < 1.0 ? norm_w * Wmax : Wmax;
    float Dw = pow(1.0 - w/Wmax, mu_plus)*fact;
    *weight_pt = w + Dw;
  }
  else {
    float fact = -Kminus*exp(Dt/tau_minus);
    //float norm_w = ((w/Wmax) - (alpha*lambda*pow(w/Wmax, mu_minus)*fact);
    //*weight_pt = norm_w > 0.0 ? norm_w * Wmax : 0.0;
    float Dw = pow(w/Wmax, mu_minus)*fact;
    *weight_pt = w + Dw;
  }
}

int STDP::Init()
{
  type_ = i_stdp_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_param_name;
  gpuErrchk(cudaMalloc(&d_param_arr_, n_param_*sizeof(float)));
  SetParam("tau_plus", 20.0);
  SetParam("tau_minus", 20.0);
  SetParam("lambda", 0.01);
  SetParam("alpha", 1.0);
  SetParam("mu_plus", 1.0);
  SetParam("mu_minus", 1.0);
  SetParam("Wmax", 100.0);
  SetParam("Kplus", 0.03);
  SetParam("Kminus", 1.035*0.03);

  return 0;
}
