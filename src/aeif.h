/*
Copyright (C) 2016 Bruno Golosio
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

#define MAX_RECEPTOR_NUM 20

__device__
void HandleSpike(float weight, int i_receptor, float *y,
		 float *params);

template<int NVAR, int NPARAMS>
__global__
void InputSpike(float weight, int i_receptor, int n, int n_par)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<ARRAY_SIZE) {
    HandleSpike(weight, i_receptor, &YArr[array_idx*NVAR],
		&ParamsArr[array_idx*NPARAMS]);
  }
}

template<int N_RECEPTORS>
int AeifUpdate(int n_receptors, int n_neurons, int it, float t1, float h_min);

template <>
int AeifUpdate<0>(int n_receptors, int n_neurons, int it, float t1,
		  float h_min);

template<int N_RECEPTORS>
int AeifUpdate(int n_receptors,
	       int n_neurons, int it, float t1, float h_min)
{
  if (N_RECEPTORS == n_receptors) {
    const int NVAR = N0_VAR + 2*N_RECEPTORS;
    const int NPARAMS = N0_PARAMS + 4*N_RECEPTORS;
    //printf("AeifUpdate nvar %d nparams %d n_neurons %d\n", NVAR, NPARAMS,
    //   n_neurons);  
    ArrayUpdate<NVAR, NPARAMS><<<(n_neurons+1023)/1024, 1024>>>(t1, h_min);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //std::cout << "AeifUpdate end\n";
    //exit(0);
  }
  else {
    AeifUpdate<N_RECEPTORS - 1>(n_receptors, n_neurons, it, t1, h_min);
  }

  return 0;
}

class AEIF
{
 public:
  RungeKutta5 rk5_;
  int n_receptors_;
  int n_neurons_;
  float h_min_;
  float h_;

  int n_var_;
  int n_params_;
  int i_node_0_;

  int Init(int i_node_0, int n_neurons, int n_receptors);

  int Calibrate(float t_min);
		
  int Update(int it, float t1);
  
  int GetX(int i_neuron, int n_neurons, float *x) {
    return rk5_.GetX(i_neuron, n_neurons, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_neurons, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_neurons, y);
  }
  
  int SetParams(std::string param_name, int i_neuron, int n_neurons,
		float val);

  int SetVectParams(std::string param_name, int i_neuron, int n_neurons,
		    float *params, int vect_size);

  int GetVarIdx(std::string var_name);

};


#endif
