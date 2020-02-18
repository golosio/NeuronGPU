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

#include <config.h>
#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "base_neuron.h"
#include "spike_buffer.h"
__global__ void BaseNeuronSetFloatArray(float *arr, int n_elem, int step,
					float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[array_idx*step] = val;
  }
}

__global__ void BaseNeuronSetFloatPtArray(float *arr, int *pos, int n_elem,
					  int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr[pos[array_idx]*step] = val;
  }
}

__global__ void BaseNeuronGetFloatArray(float *arr1, float *arr2, int n_elem,
					int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[array_idx*step1];
  }
}

__global__ void BaseNeuronGetFloatPtArray(float *arr1, float *arr2, int *pos,
					  int n_elem, int step1, int step2)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elem) {
    arr2[array_idx*step2] = arr1[pos[array_idx]*step1];
  }
}

int BaseNeuron::Init(int i_node_0, int n_node, int n_port,
		     int i_group, unsigned long long *seed)
{
  node_type_= 0; // NULL MODEL
  i_node_0_ = i_node_0;
  n_node_ = n_node;
  n_port_ = n_port;
  i_group_ = i_group;
  seed_ = seed;
  
  n_scal_var_ = 0;
  n_port_var_ = 0;
  n_scal_param_ = 0;
  n_port_param_ = 0;
  n_var_ = 0;
  n_param_ = 0;
  n_array_var_ = 0;
  n_array_param_ = 0;

  get_spike_array_ = NULL;
  port_weight_arr_ = NULL;
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;
  port_input_arr_ = NULL;
  port_input_arr_step_ = 0;
  port_input_port_step_ = 0;
  var_arr_ = NULL;
  param_arr_ = NULL;
  scal_var_name_ = NULL;
  port_var_name_= NULL;
  scal_param_name_ = NULL;
  port_param_name_ = NULL;
  array_var_name_= NULL;
  array_param_name_ = NULL;

  d_dir_conn_array_ = NULL;
  n_dir_conn_ = 0;
  has_dir_conn_ = false;
 
  return 0;
}			    

int BaseNeuron::SetScalParam(int i_neuron, int n_neuron,
			     std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);
  BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, n_neuron, n_param_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int BaseNeuron::SetScalParam(int *i_neuron, int n_neuron,
			     std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);
  BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_i_neuron, n_neuron, n_param_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_i_neuron));
  
  return 0;
}

int BaseNeuron::SetPortParam(int i_neuron, int n_neuron,
			     std::string param_name, float *param,
			     int vect_size)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  if (vect_size != n_port_) {
    throw ngpu_exception("Parameter array size must be equal "
			 "to the number of ports.");
  }
  float *param_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    param_pt = GetParamPt(i_neuron, param_name, i_vect);
    BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, n_neuron, n_param_, param[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

int BaseNeuron::SetPortParam(int *i_neuron, int n_neuron,
			     std::string param_name, float *param,
			     int vect_size)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  if (vect_size != n_port_) {
    throw ngpu_exception("Parameter array size must be equal "
			 "to the number of ports.");
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *param_pt = GetParamPt(0, param_name, i_vect);
    BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_i_neuron, n_neuron, n_param_, param[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  gpuErrchk(cudaFree(d_i_neuron));

  return 0;
}

int BaseNeuron::SetArrayParam(int i_neuron, int n_neuron,
			      std::string param_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

int BaseNeuron::SetArrayParam(int *i_neuron, int n_neuron,
			      std::string param_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

int BaseNeuron::SetScalVar(int i_neuron, int n_neuron,
			     std::string var_name, float val)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt = GetVarPt(i_neuron, var_name);
  BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, n_neuron, n_var_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int BaseNeuron::SetScalVar(int *i_neuron, int n_neuron,
			   std::string var_name, float val)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
				     + var_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *var_pt = GetVarPt(0, var_name);
  BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_i_neuron, n_neuron, n_var_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_i_neuron));
  
  return 0;
}

int BaseNeuron::SetPortVar(int i_neuron, int n_neuron,
			   std::string var_name, float *var,
			   int vect_size)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  if (vect_size != n_port_) {
    throw ngpu_exception("Variable array size must be equal "
			 "to the number of ports.");
  }
  float *var_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    var_pt = GetVarPt(i_neuron, var_name, i_vect);
    BaseNeuronSetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, n_neuron, n_var_, var[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

int BaseNeuron::SetPortVar(int *i_neuron, int n_neuron,
			   std::string var_name, float *var,
			   int vect_size)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  if (vect_size != n_port_) {
    throw ngpu_exception("Variable array size must be equal "
			 "to the number of ports.");
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *var_pt = GetVarPt(0, var_name, i_vect);
    BaseNeuronSetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_i_neuron, n_neuron, n_var_, var[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  gpuErrchk(cudaFree(d_i_neuron));

  return 0;
}

int BaseNeuron::SetArrayVar(int i_neuron, int n_neuron,
			      std::string var_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}

int BaseNeuron::SetArrayVar(int *i_neuron, int n_neuron,
			      std::string var_name, float *array,
			      int array_size)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}

float *BaseNeuron::GetScalParam(int i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);

  float *d_param_arr;
  gpuErrchk(cudaMalloc(&d_param_arr, n_neuron*sizeof(float)));
  float *h_param_arr = (float*)malloc(n_neuron*sizeof(float));

  BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_param_arr, n_neuron, n_param_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_param_arr));
  
  return h_param_arr;
}

float *BaseNeuron::GetScalParam(int *i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);

  float *d_param_arr;
  gpuErrchk(cudaMalloc(&d_param_arr, n_neuron*sizeof(float)));
  float *h_param_arr = (float*)malloc(n_neuron*sizeof(float));
  
  BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (param_pt, d_param_arr, d_i_neuron, n_neuron, n_param_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_i_neuron));

  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_param_arr));

  return h_param_arr;
}

float *BaseNeuron::GetPortParam(int i_neuron, int n_neuron,
			      std::string param_name)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *param_pt;

  float *d_param_arr;
  gpuErrchk(cudaMalloc(&d_param_arr, n_neuron*n_port_*sizeof(float)));
  float *h_param_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
  
  for (int port=0; port<n_port_; port++) {
    param_pt = GetParamPt(i_neuron, param_name, port);
    BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_param_arr + port, n_neuron, n_param_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_param_arr));
  
  return h_param_arr;
}

float *BaseNeuron::GetPortParam(int *i_neuron, int n_neuron,
				std::string param_name)
{
  if (!IsPortParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));

  float *d_param_arr;
  gpuErrchk(cudaMalloc(&d_param_arr, n_neuron*n_port_*sizeof(float)));
  float *h_param_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
    
  for (int port=0; port<n_port_; port++) {
    float *param_pt = GetParamPt(0, param_name, port);
    BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (param_pt, d_param_arr+port, d_i_neuron, n_neuron, n_param_,
       n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  gpuErrchk(cudaFree(d_i_neuron));
  
  gpuErrchk(cudaMemcpy(h_param_arr, d_param_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_param_arr));
  
  return h_param_arr;
}

float *BaseNeuron::GetArrayParam(int i_neuron, std::string param_name)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);
}

float *BaseNeuron::GetScalVar(int i_neuron, int n_neuron,
				std::string var_name)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt = GetVarPt(i_neuron, var_name);

  float *d_var_arr;
  gpuErrchk(cudaMalloc(&d_var_arr, n_neuron*sizeof(float)));
  float *h_var_arr = (float*)malloc(n_neuron*sizeof(float));

  BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, n_neuron, n_var_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_var_arr));
  
  return h_var_arr;
}

float *BaseNeuron::GetScalVar(int *i_neuron, int n_neuron,
				std::string var_name)
{
  if (!IsScalVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
				     + var_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *var_pt = GetVarPt(0, var_name);

  float *d_var_arr;
  gpuErrchk(cudaMalloc(&d_var_arr, n_neuron*sizeof(float)));
  float *h_var_arr = (float*)malloc(n_neuron*sizeof(float));
  
  BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
    (var_pt, d_var_arr, d_i_neuron, n_neuron, n_var_, 1);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_i_neuron));

  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*sizeof(float),
		       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_var_arr));

  return h_var_arr;
}

float *BaseNeuron::GetPortVar(int i_neuron, int n_neuron,
			      std::string var_name)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neuron - 1);
  float *var_pt;

  float *d_var_arr;
  gpuErrchk(cudaMalloc(&d_var_arr, n_neuron*n_port_*sizeof(float)));
  float *h_var_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
  
  for (int port=0; port<n_port_; port++) {
    var_pt = GetVarPt(i_neuron, var_name, port);
    BaseNeuronGetFloatArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_var_arr + port, n_neuron, n_var_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_var_arr));
  
  return h_var_arr;
}

float *BaseNeuron::GetPortVar(int *i_neuron, int n_neuron,
			      std::string var_name)
{
  if (!IsPortVar(var_name)) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
			 + var_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neuron*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neuron*sizeof(int),
		       cudaMemcpyHostToDevice));

  float *d_var_arr;
  gpuErrchk(cudaMalloc(&d_var_arr, n_neuron*n_port_*sizeof(float)));
  float *h_var_arr = (float*)malloc(n_neuron*n_port_*sizeof(float));
    
  for (int port=0; port<n_port_; port++) {
    float *var_pt = GetVarPt(0, var_name, port);
    BaseNeuronGetFloatPtArray<<<(n_neuron+1023)/1024, 1024>>>
      (var_pt, d_var_arr+port, d_i_neuron, n_neuron, n_var_, n_port_);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  gpuErrchk(cudaFree(d_i_neuron));
  
  gpuErrchk(cudaMemcpy(h_var_arr, d_var_arr, n_neuron*n_port_
		       *sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_var_arr));
  
  return h_var_arr;
}

float *BaseNeuron::GetArrayVar(int i_neuron, std::string var_name)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);
}

int BaseNeuron::GetScalVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) break;
  }
  if (i_var == n_scal_var_) {
    throw ngpu_exception(std::string("Unrecognized scalar variable ")
			 + var_name);
  }
  
  return i_var;
}

int BaseNeuron::GetPortVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_port_var_; i_var++) {
    if (var_name == port_var_name_[i_var]) break;
  }
  if (i_var == n_port_var_) {
    throw ngpu_exception(std::string("Unrecognized port variable ")
				     + var_name);
  }
  
  return i_var;
}

int BaseNeuron::GetScalParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_param_; i_param++) {
    if (param_name == scal_param_name_[i_param]) break;
  }
  if (i_param == n_scal_param_) {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
  
  return i_param;
}

int BaseNeuron::GetPortParamIdx(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_port_param_; i_param++) {
    if (param_name == port_param_name_[i_param]) break;
  }
  if (i_param == n_port_param_) {
    throw ngpu_exception(std::string("Unrecognized port parameter ")
			 + param_name);
  }
  
  return i_param;
}

float *BaseNeuron::GetVarArr()
{
  return var_arr_;
}

float *BaseNeuron::GetParamArr()
{
  return param_arr_;
}


int BaseNeuron::GetArrayVarSize(int i_neuron, std::string var_name)
{
  throw ngpu_exception(std::string("Unrecognized variable ")
		       + var_name);

}
  
int BaseNeuron::GetArrayParamSize(int i_neuron, std::string param_name)
{
  throw ngpu_exception(std::string("Unrecognized parameter ")
		       + param_name);

}

int BaseNeuron::GetVarSize(std::string var_name)
{
  if (IsScalVar(var_name)) {
    return 1;
  }
  else if (IsPortVar(var_name)) {
    return n_port_;
  }
  else if (IsArrayVar(var_name)) {
    throw ngpu_exception(std::string("Node index must be specified to get "
				     "array variable size for ")+ var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

int BaseNeuron::GetParamSize(std::string param_name)
{
  if (IsScalParam(param_name)) {
    return 1;
  }
  else if (IsPortParam(param_name)) {
    return n_port_;
  }
  else if (IsArrayParam(param_name)) {
    throw ngpu_exception(std::string("Node index must be specified to get "
				     "array parameter size for ")+ param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}


bool BaseNeuron::IsScalVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) return true;
  }
  return false;
}

bool BaseNeuron::IsPortVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_port_var_; i_var++) {
    if (var_name == port_var_name_[i_var]) return true;
  }
  return false;
}

bool BaseNeuron::IsArrayVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_array_var_; i_var++) {
    if (var_name == array_var_name_[i_var]) return true;
  }
  return false;
}

bool BaseNeuron::IsScalParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_param_; i_param++) {
    if (param_name == scal_param_name_[i_param]) return true;
  }
  return false;
}

bool BaseNeuron::IsPortParam(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_port_param_; i_param++) {
    if (param_name == port_param_name_[i_param]) return true;
  }
  return false;
}

bool BaseNeuron::IsArrayParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_array_param_; i_param++) {
    if (param_name == array_param_name_[i_param]) return true;
  }
  return false;
}

int BaseNeuron::CheckNeuronIdx(int i_neuron)
{
  if (i_neuron>=n_node_) {
    throw ngpu_exception("Neuron index must be lower then n. of neurons");
  }
  else if (i_neuron<0) {
    throw ngpu_exception("Neuron index must be >= 0");
  }
  return 0;
}

int BaseNeuron::CheckPortIdx(int port)
{
  if (port>=n_port_) {
    throw ngpu_exception("Port index must be lower then n. of ports");
  }
  else if (port<0) {
    throw ngpu_exception("Port index must be >= 0");
  }
  return 0;
}

float *BaseNeuron::GetVarPt(int i_neuron, std::string var_name,
			    int port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (port!=0) {
    CheckPortIdx(port);
  }
    
  if (IsScalVar(var_name)) {
    int i_var =  GetScalVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + i_var;
  }
  else if (IsPortVar(var_name)) {
    int i_vvar =  GetPortVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + n_scal_var_
      + port*n_port_var_ + i_vvar;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *BaseNeuron::GetParamPt(int i_neuron, std::string param_name,
			      int port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (port!=0) {
    CheckPortIdx(port);
  }
  if (IsScalParam(param_name)) {
    int i_param =  GetScalParamIdx(param_name);
    return GetParamArr() + i_neuron*n_param_ + i_param;
  }
  else if (IsPortParam(param_name)) {
    int i_vparam =  GetPortParamIdx(param_name);
    return GetParamArr() + i_neuron*n_param_ + n_scal_param_
      + port*n_port_param_ + i_vparam;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float BaseNeuron::GetSpikeActivity(int i_neuron)
{
  int i_spike_buffer = i_neuron + i_node_0_;
  int Ns;
  gpuErrchk(cudaMemcpy(&Ns, d_SpikeBufferSize + i_spike_buffer,
		       sizeof(int), cudaMemcpyDeviceToHost));
  if (Ns==0) {
    return 0.0;
  }
  int time_idx;
  // get first (most recent) spike from buffer
  gpuErrchk(cudaMemcpy(&time_idx, d_SpikeBufferTimeIdx + i_spike_buffer,
		       sizeof(int), cudaMemcpyDeviceToHost));
  if (time_idx!=0) { // neuron is not spiking now
    return 0.0;
  }
  float spike_height;
  gpuErrchk(cudaMemcpy(&spike_height, d_SpikeBufferHeight + i_spike_buffer,
		       sizeof(float), cudaMemcpyDeviceToHost));

  return spike_height;
}

std::vector<std::string> BaseNeuron::GetScalVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_scal_var_; i++) {
    var_name_vect.push_back(scal_var_name_[i]);
  }
  
  return var_name_vect;
}
  
int BaseNeuron::GetNScalVar()
{
  return n_scal_var_;
}

std::vector<std::string> BaseNeuron::GetPortVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_port_var_; i++) {
    var_name_vect.push_back(port_var_name_[i]);
  }
  
  return var_name_vect;
}
  
int BaseNeuron::GetNPortVar()
{
  return n_port_var_;
}

std::vector<std::string> BaseNeuron::GetScalParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_scal_param_; i++) {
    param_name_vect.push_back(scal_param_name_[i]);
  }
  
  return param_name_vect;
}
  
int BaseNeuron::GetNScalParam()
{
  return n_scal_param_;
}

std::vector<std::string> BaseNeuron::GetPortParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_port_param_; i++) {
    param_name_vect.push_back(port_param_name_[i]);
  }
  
  return param_name_vect;
}
  
int BaseNeuron::GetNPortParam()
{
  return n_port_param_;
}


std::vector<std::string> BaseNeuron::GetArrayVarNames()
{
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_array_var_; i++) {
    var_name_vect.push_back(array_var_name_[i]);
  }
  
  return var_name_vect;
}
  
int BaseNeuron::GetNArrayVar()
{
  return n_array_var_;
}

std::vector<std::string> BaseNeuron::GetArrayParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_array_param_; i++) {
    param_name_vect.push_back(array_param_name_[i]);
  }
  
  return param_name_vect;
}
  
int BaseNeuron::GetNArrayParam()
{
  return n_array_param_;
}
