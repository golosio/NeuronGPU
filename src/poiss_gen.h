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

#ifndef POISSGENH
#define POISSGENH

#include <iostream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

/*
const int N_POISS_GEN_SCAL_PARAM = 4;
const std::string poiss_gen_scal_param_name[] = {
  "rate",
  "origin"
  "start",
  "stop",
};
*/

class poiss_gen : public BaseNeuron
{
  curandState *d_curand_state_;
 public:
  
  int Init(int i_node_0, int n_node, int n_port, int i_group,
	   unsigned long long *seed);

  int Calibrate(double, float);
		
  int Update(long long it, double t1);
  int SendDirectSpikes(double t, float time_step);

};


#endif
