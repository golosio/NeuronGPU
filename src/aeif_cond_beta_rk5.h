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

#ifndef AEIFCONDBETARK5H
#define AEIFCONDBETARK5H

#include <string>
#include <stdio.h>
#include <math.h>
#include "node_group.h"

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void aeif_cond_beta_Derivatives(float x, float *y, float *dydx, float *params,
				DataStruct data_struct);

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void aeif_cond_beta_ExternalUpdate(float x, float *y, float *params,
				   bool end_time_step,
				   RK5DataStruct data_struct);

template<class DataStruct>
__device__
void aeif_cond_beta_NodeInit(int n_var, int n_params, float x, float *y,
			     float *params, DataStruct data_struct);


template<class DataStruct>
__device__
void aeif_cond_beta_NodeCalibrate(int n_var, int n_params, float x, float *y,
				  float *params, DataStruct data_struct);


#endif
