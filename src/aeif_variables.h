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

#ifndef AEIFVARIABLESH
#define AEIFVARIABLESH

#include <string>

enum ScalVarIndexes {
  i_V_m = 0,
  i_w,
  N_SCAL_VAR
};

enum VectVarIndexes {
  i_g = 0,
  i_g1,
  N_VECT_VAR
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

enum VectParamIndexes {
  i_E_rev = 0,
  i_taus_rise,
  i_taus_decay,
  i_g0,
  N_VECT_PARAMS
};

const std::string aeif_scal_var_names[N_SCAL_VAR] = {
  "V_m",
  "w"
};

const std::string aeif_vect_var_names[N_VECT_VAR] = {
  "g",
  "g1"
};

const std::string aeif_scal_param_names[N_SCAL_PARAMS] = {
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

const std::string aeif_vect_param_names[N_VECT_PARAMS] = {
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
#define g(i) y[N_SCAL_VAR + N_VECT_VAR*i + i_g]
#define g1(i) y[N_SCAL_VAR + N_VECT_VAR*i + i_g1]

#define dVdt dydx[i_V_m]
#define dwdt dydx[i_w]
#define dgdt(i) dydx[N_SCAL_VAR + N_VECT_VAR*i + i_g]
#define dg1dt(i) dydx[N_SCAL_VAR + N_VECT_VAR*i + i_g1]

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

#define E_rev(i) params[N_SCAL_PARAMS + N_VECT_PARAMS*i + i_E_rev]
#define taus_rise(i) params[N_SCAL_PARAMS + N_VECT_PARAMS*i + i_taus_rise]
#define taus_decay(i) params[N_SCAL_PARAMS + N_VECT_PARAMS*i + i_taus_decay]
#define g0(i) params[N_SCAL_PARAMS + N_VECT_PARAMS*i + i_g0]

#endif
