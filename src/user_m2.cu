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

#include <math.h>
#include <iostream>
#include "user_m2_variables.h"
#include "rk5.h"
#include "user_m2.h"

using namespace user_m2_ns;

int user_m2::Init(int i_node_0, int n_node, int n_port,
			 int i_group, unsigned long long *seed) {
  BaseNeuron::Init(i_node_0, n_node, n_port, i_group, seed);
  h_min_=1.0e-4;
  h_ = 1.0e-2;
  node_type_ = i_user_m2_model;
  n_scal_var_ = N_SCAL_VAR;
  n_port_var_ = N_PORT_VAR;
  n_scal_param_ = N_SCAL_PARAM;
  n_port_param_ = N_PORT_PARAM;

  n_var_ = n_scal_var_ + n_port_var_*n_port;
  n_param_ = n_scal_param_ + n_port_param_*n_port;

  scal_var_name_ = user_m2_scal_var_name;
  port_var_name_= user_m2_port_var_name;
  scal_param_name_ = user_m2_scal_param_name;
  port_param_name_ = user_m2_port_param_name;
  rk5_data_struct_.node_type_ = i_user_m2_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  rk5_.Init(n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_);
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  port_weight_arr_ = GetParamArr() + n_scal_param_
    + GetPortParamIdx("g0");
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = n_port_param_;

  port_input_arr_ = GetVarArr() + n_scal_var_
    + GetPortVarIdx("g1");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_port_var_;

  return 0;
}

int user_m2::Calibrate(float time_min, float /*time_resolution*/)
{
  rk5_.Calibrate(time_min, h_, rk5_data_struct_);
  
  return 0;
}

template <>
int user_m2::UpdateNR<0>(int it, float t1)
{
  return 0;
}

int user_m2::Update(int it, float t1) {
  UpdateNR<MAX_PORT_NUM>(it, t1);

  return 0;
}

