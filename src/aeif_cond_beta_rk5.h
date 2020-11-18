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

struct aeif_cond_beta_rk5;


template<int NVAR, int NPARAM>
__device__
void Derivatives(double x, float *y, float *dydx, float *param,
		 aeif_cond_beta_rk5 data_struct);

template<int NVAR, int NPARAM>
__device__
void ExternalUpdate(double x, float *y, float *param, bool end_time_step,
		    aeif_cond_beta_rk5 data_struct);

__device__
void NodeInit(int n_var, int n_param, double x, float *y,
	      float *param, aeif_cond_beta_rk5 data_struct);

__device__
void NodeCalibrate(int n_var, int n_param, double x, float *y,
		   float *param, aeif_cond_beta_rk5 data_struct);

#endif
