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

#ifndef RK5INTERFACEH
#define RK5INTERFACEH

#include <stdio.h>
#include <math.h>

#include "neuron_models.h"
#include "aeif_cond_beta_rk5.h"
#include "user_m1_rk5.h"
#include "user_m2_rk5.h"

				    //#include "aeif_cond_beta_variables.h"

template<int NVAR, int NPARAM, class DataStruct>
__device__
    void Derivatives(float x, float *y, float *dydx, float *param,
		     RK5DataStruct data_struct)
{
  switch (data_struct.node_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_ns::Derivatives(x, y, dydx, NVAR, param,
				   data_struct);
    break;
  case i_user_m1_model:
    user_m1_ns::Derivatives(x, y, dydx, NVAR, param,
			    data_struct);
    break;
  case i_user_m2_model:
    user_m2_ns::Derivatives(x, y, dydx, NVAR, param,
			    data_struct);
    break;
  }
}

template<int NVAR, int NPARAM, class DataStruct>
__device__
    void ExternalUpdate
    (float x, float *y, float *param, bool end_time_step,
			RK5DataStruct data_struct)
{
  switch (data_struct.node_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_ns::ExternalUpdate(x, y, param,
				      end_time_step,
				      data_struct);
    break;
  case i_user_m1_model:
    user_m1_ns::ExternalUpdate(x, y, param,
			       end_time_step,
			       data_struct);
    break;
  case i_user_m2_model:
    user_m2_ns::ExternalUpdate(x, y, param,
			       end_time_step,
			       data_struct);
    break;
  }    
}


template<class DataStruct>
__device__
void NodeInit(int n_var, int n_param, float x, float *y,
	     float *param, DataStruct data_struct)
{
  switch (data_struct.node_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_ns::NodeInit(n_var, n_param, x, y, param,
				data_struct);
    break;
  case i_user_m1_model:
    user_m1_ns::NodeInit(n_var, n_param, x, y, param,
				data_struct);
    break;
  case i_user_m2_model:
    user_m2_ns::NodeInit(n_var, n_param, x, y, param,
				data_struct);
    break;
  }
}

template<class DataStruct>
__device__
void NodeCalibrate(int n_var, int n_param, float x, float *y,
		  float *param, DataStruct data_struct)

{
  switch (data_struct.node_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
    break;
  case i_user_m1_model:
    user_m1_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
    break;
  case i_user_m2_model:
    user_m2_ns::NodeCalibrate(n_var, n_param, x, y, param, data_struct);
    break;
  }
}

#endif
