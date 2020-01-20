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

#include <iostream>
#include "ngpu_exception.h"
#include "cuda_error.h"
#include "base_neuron.h"
#include "spike_buffer.h"
__global__ void BaseNeuronSetFloatArray(float *arr, int n_elems, int step,
					float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elems) {
    arr[array_idx*step] = val;
  }
}

__global__ void BaseNeuronSetFloatPtArray(float *arr, int *pos, int n_elems,
					  int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elems) {
    arr[pos[array_idx]*step] = val;
  }
}

int BaseNeuron::Init(int i_node_0, int n_nodes, int n_ports,
		     int i_group, unsigned long long *seed)
{
  node_type_= 0; // NULL MODEL
  i_node_0_ = i_node_0;
  n_nodes_ = n_nodes;
  n_ports_ = n_ports;
  i_group_ = i_group;
  seed_ = seed;
  
  n_scal_var_ = 0;
  n_vect_var_ = 0;
  n_scal_params_ = 0;
  n_vect_params_ = 0;
  n_var_ = 0;
  n_params_ = 0;

  get_spike_array_ = NULL;
  port_weight_arr_ = NULL;
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;
  port_input_arr_ = NULL;
  port_input_arr_step_ = 0;
  port_input_port_step_ = 0;
  var_arr_ = NULL;
  params_arr_ = NULL;
  scal_var_name_ = NULL;
  vect_var_name_= NULL;
  scal_param_name_ = NULL;
  vect_param_name_ = NULL;
  d_dir_conn_array_ = NULL;
  n_dir_conn_ = 0;
  has_dir_conn_ = false;
 
  return 0;
}			    

int BaseNeuron::SetScalParam(int i_neuron, int n_neurons,
			     std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neurons - 1);
  float *param_pt = GetParamPt(i_neuron, param_name);
  BaseNeuronSetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
    (param_pt, n_neurons, n_params_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int BaseNeuron::SetScalParam( int *i_neuron, int n_neurons,
			      std::string param_name, float val)
{
  if (!IsScalParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized scalar parameter ")
				     + param_name);
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neurons*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neurons*sizeof(int),
		       cudaMemcpyHostToDevice));
  float *param_pt = GetParamPt(0, param_name);
  BaseNeuronSetFloatPtArray<<<(n_neurons+1023)/1024, 1024>>>
    (param_pt, d_i_neuron, n_neurons, n_params_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_i_neuron));
  
  return 0;
}

int BaseNeuron::SetVectParam( int i_neuron, int n_neurons,
			      std::string param_name, float *params,
			      int vect_size)
{
  if (!IsVectParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized vector parameter ")
			 + param_name);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neurons - 1);
  if (vect_size != n_ports_) {
    throw ngpu_exception("Parameter vector size must be equal "
			 "to the number of ports.");
  }
  float *param_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    param_pt = GetParamPt(i_neuron, param_name, i_vect);
    BaseNeuronSetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
      (param_pt, n_neurons, n_params_, params[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  return 0;
}

int BaseNeuron::SetVectParam( int *i_neuron, int n_neurons,
			      std::string param_name, float *params,
			      int vect_size)
{
  if (!IsVectParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized vector parameter ")
			 + param_name);
  }
  if (vect_size != n_ports_) {
    throw ngpu_exception("Parameter vector size must be equal "
			 "to the number of ports.");
  }
  int *d_i_neuron;
  gpuErrchk(cudaMalloc(&d_i_neuron, n_neurons*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_i_neuron, i_neuron, n_neurons*sizeof(int),
		       cudaMemcpyHostToDevice));
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    float *param_pt = GetParamPt(0, param_name, i_vect);
    BaseNeuronSetFloatPtArray<<<(n_neurons+1023)/1024, 1024>>>
      (param_pt, d_i_neuron, n_neurons, n_params_, params[i_vect]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  gpuErrchk(cudaFree(d_i_neuron));

  return 0;
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

int BaseNeuron::GetVectVarIdx(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_vect_var_; i_var++) {
    if (var_name == vect_var_name_[i_var]) break;
  }
  if (i_var == n_vect_var_) {
    throw ngpu_exception(std::string("Unrecognized vector variable ")
				     + var_name);
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
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
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
    throw ngpu_exception(std::string("Unrecognized vector parameter ")
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
  return params_arr_;
}

bool BaseNeuron::IsScalVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_scal_var_; i_var++) {
    if (var_name == scal_var_name_[i_var]) return true;
  }
  return false;
}

bool BaseNeuron::IsVectVar(std::string var_name)
{
  int i_var;
  for (i_var=0; i_var<n_vect_var_; i_var++) {
    if (var_name == vect_var_name_[i_var]) return true;
  }
  return false;
}

bool BaseNeuron::IsScalParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_scal_params_; i_param++) {
    if (param_name == scal_param_name_[i_param]) return true;
  }
  return false;
}

bool BaseNeuron::IsVectParam(std::string param_name)
{  
  int i_param;
  for (i_param=0; i_param<n_vect_params_; i_param++) {
    if (param_name == vect_param_name_[i_param]) return true;
  }
  return false;
}

int BaseNeuron::CheckNeuronIdx(int i_neuron)
{
  if (i_neuron>=n_nodes_) {
    throw ngpu_exception("Neuron index must be lower then n. of neurons");
  }
  else if (i_neuron<0) {
    throw ngpu_exception("Neuron index must be >= 0");
  }
  return 0;
}

int BaseNeuron::CheckPortIdx(int i_port)
{
  if (i_port>=n_ports_) {
    throw ngpu_exception("Port index must be lower then n. of ports");
  }
  else if (i_port<0) {
    throw ngpu_exception("Port index must be >= 0");
  }
  return 0;
}

float *BaseNeuron::GetVarPt(int i_neuron, std::string var_name,
			    int i_port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (i_port!=0) {
    CheckPortIdx(i_port);
  }
    
  if (IsScalVar(var_name)) {
    int i_var =  GetScalVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + i_var;
  }
  else if (IsVectVar(var_name)) {
    int i_vvar =  GetVectVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + n_scal_var_
      + i_port*n_vect_var_ + i_vvar;
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *BaseNeuron::GetParamPt(int i_neuron, std::string param_name,
			      int i_port /*=0*/)
{
  CheckNeuronIdx(i_neuron);
  if (i_port!=0) {
    CheckPortIdx(i_port);
  }
  if (IsScalParam(param_name)) {
    int i_param =  GetScalParamIdx(param_name);
    return GetParamArr() + i_neuron*n_params_ + i_param;
  }
  else if (IsVectParam(param_name)) {
    int i_vparam =  GetVectParamIdx(param_name);
    return GetParamArr() + i_neuron*n_params_ + n_scal_params_
      + i_port*n_vect_params_ + i_vparam;
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

