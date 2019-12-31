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

#include <iostream>
#include "cuda_error.h"
#include "base_neuron.h"

__global__ void SetFloatArray(float *arr, int n_elems, int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elems) {
    arr[array_idx*step] = val;
  }
}

int BaseNeuron::SetScalParams(std::string param_name, int i_neuron,
		    int n_neurons, float val) {

  int i_param;
  for (i_param=0; i_param<n_scal_params_; i_param++) {
    if (param_name == scal_param_name_[i_param]) break;
  }
  if (i_param == n_scal_params_) {
    std::cerr << "Unrecognized parameter " << param_name << " .\n";
    exit(-1);
  }

  SetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
    (&params_arr_[i_neuron*n_params_ + i_param], n_neurons, n_params_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int BaseNeuron::SetVectParams(std::string param_name, int i_neuron,
			      int n_neurons, float *params, int vect_size) {
  int i_vect;
  for (i_vect=0; i_vect<n_vect_params_; i_vect++) {
    if (param_name == vect_param_name_[i_vect]) break;
  }
  if (i_vect == n_vect_params_) {
    std::cerr << "Unrecognized vector parameter " << param_name << " \n";
    exit(-1);
  }  
  if (vect_size != n_receptors_) {
    std::cerr << "Parameter vector size must be equal to the number "
      "of receptor ports.\n";
    exit(-1);
  }  

  for (int i=0; i<vect_size; i++) {
    int i_param = n_scal_params_ + n_vect_params_*i + i_vect;
    SetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
      (&params_arr_[i_neuron*n_params_ + i_param], n_neurons, n_params_,
       params[i]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

int BaseNeuron::GetScalVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) break;
  }
  if (i_var == n_scal_var_) {
    std::cerr << "Unrecognized variable " << var_name << " .\n";
    exit(-1);
  }
  
  return i_var;
}

int BaseNeuron::GetVectVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_vect_var_; i_var++) {
    if (var_name == vect_var_name_[i_var]) break;
  }
  if (i_var == n_vect_var_) {
    std::cerr << "Unrecognized variable " << var_name << " .\n";
    exit(-1);
  }
  
  return i_var;
}

int BaseNeuron::GetScalParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_params_; i_param++) {
    if (param_name == scal_param_name_[i_param]) break;
  }
  if (i_param == n_scal_params_) {
    std::cerr << "Unrecognized parameter " << param_name << " .\n";
    exit(-1);
  }
  
  return i_param;
}

int BaseNeuron::GetVectParamIdx(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_vect_params_; i_param++) {
    if (param_name == vect_param_name_[i_param]) break;
  }
  if (i_param == n_vect_params_) {
    std::cerr << "Unrecognized vector parameter " << param_name << " .\n";
    exit(-1);
  }
  
  return i_param;
}

float *BaseNeuron::GetVarArr()
{
  return var_arr_;
}

float *BaseNeuron::GetParamsArr()
{
  return params_arr_;
}
