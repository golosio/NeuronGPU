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

#ifndef AEIFH
#define AEIFH
#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "neuron_group.h"
#include "base_neuron.h"
#include "neuron_models.h"
#define MAX_RECEPTOR_NUM 20

class AEIF : public BaseNeuron
{
 public:
  RungeKutta5<RK5DataStruct> rk5_;
  float h_min_;
  float h_;
    
  int Init(int i_node_0, int n_neurons, int n_receptors, int i_neuron_group);

  int Calibrate(float t_min);
		
  int Update(int it, float t1);
  
  int GetX(int i_neuron, int n_neurons, float *x) {
    return rk5_.GetX(i_neuron, n_neurons, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_neurons, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_neurons, y);
  }
  
  template<int N_RECEPTORS>
    int UpdateNR(int it, float t1);

};

template <>
int AEIF::UpdateNR<0>(int it, float t1);

template<int N_RECEPTORS>
int AEIF::UpdateNR(int it, float t1)
{
  if (N_RECEPTORS == n_receptors_) {
    const int NVAR = N_SCAL_VAR + N_VECT_VAR*N_RECEPTORS;
    const int NPARAMS = N_SCAL_PARAMS + N_VECT_PARAMS*N_RECEPTORS;

    RK5DataStruct data_struct = {i_AEIF_model, i_node_0_};
    rk5_.Update<NVAR, NPARAMS>(t1, h_min_, data_struct);
  }
  else {
    UpdateNR<N_RECEPTORS - 1>(it, t1);
  }

  return 0;
}



#endif
