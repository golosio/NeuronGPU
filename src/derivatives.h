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

#ifndef DERIVATIVESH
#define DERIVATIVESH

#include <stdio.h>
#include <math.h>
#include "aeif_derivatives.h"
#include "neuron_models.h"

template<int NVAR, int NPARAMS, class DataStruct>
__device__
    void Derivatives(float x, float *y, float *dydx, float *params,
		     RK5DataStruct data_struct)
{
  switch (data_struct.neuron_type_) {
  case i_AEIF_model:
    AEIF_Derivatives<NVAR, NPARAMS, DataStruct>(x, y, dydx, params,
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
  case i_AEIF_model:
    AEIF_ExternalUpdate<NVAR, NPARAMS, DataStruct>(x, y, params, end_time_step,
						 data_struct);
    break;
  }    
}

#endif
