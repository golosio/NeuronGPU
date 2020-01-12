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

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void Derivatives(float x, float *y, float *dydx, float *params,
		     RK5DataStruct data_struct)
{
  switch (data_struct.neuron_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_Derivatives<NVAR, NPARAMS, DataStruct>(x, y, dydx, params,
						data_struct);
    break;
  }
}

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void ExternalUpdate
    (float x, float *y, float *params, bool end_time_step,
			RK5DataStruct data_struct)
{
  switch (data_struct.neuron_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_ExternalUpdate<NVAR, NPARAMS, DataStruct>(x, y, params,
							     end_time_step,
							     data_struct);
    break;
  }    
}


template<class DataStruct>
__device__
void NodeInit(int n_var, int n_params, float x, float *y,
	     float *params, DataStruct data_struct)
{
  switch (data_struct.neuron_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_NodeInit<DataStruct>(n_var, n_params, x, y, params,
					data_struct);
    break;
  }
}

template<class DataStruct>
__device__
void NodeCalibrate(int n_var, int n_params, float x, float *y,
		  float *params, DataStruct data_struct)

{
  switch (data_struct.neuron_type_) {
  case i_aeif_cond_beta_model:
    aeif_cond_beta_NodeCalibrate<DataStruct>(n_var, n_params, x, y, params, data_struct);
    break;
  }
}

#endif
