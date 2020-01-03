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

int BaseNeuron::Init(int i_node_0, int n_neurons, int n_receptors,
		   int i_neuron_group)
{
  i_node_0_ = i_node_0;
  n_neurons_ = n_neurons;
  n_receptors_ = n_receptors;
  i_neuron_group_ = i_neuron_group;

  return 0;
}			    
int BaseNeuron::SetScalParams(std::string param_name, int i_neuron,
		    int n_neurons, float val) {
  if (!IsScalParam(param_name)) {
    std::cerr << "Unrecognized scalar parameter " << param_name << " \n";
    exit(-1);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neurons - 1);
  float *param_pt = GetParamPt(param_name, i_neuron, 0);
  BaseNeuronSetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
    (param_pt, n_neurons, n_params_, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int BaseNeuron::SetVectParams(std::string param_name, int i_neuron,
			      int n_neurons, float *params, int vect_size) {
  if (!IsVectParam(param_name)) {
    std::cerr << "Unrecognized vector parameter " << param_name << " \n";
    exit(-1);
  }
  CheckNeuronIdx(i_neuron);
  CheckNeuronIdx(i_neuron + n_neurons - 1);
  if (vect_size != n_receptors_) {
    std::cerr << "Parameter vector size must be equal to the number "
      "of receptor ports.\n";
    exit(-1);
  }
  float *param_pt;
    
  for (int i_vect=0; i_vect<vect_size; i_vect++) {
    param_pt = GetParamPt(param_name, i_neuron, i_vect);
    BaseNeuronSetFloatArray<<<(n_neurons+1023)/1024, 1024>>>
      (param_pt, n_neurons, n_params_, params[i_vect]);
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
    std::cerr << "Unrecognized scalar variable " << var_name << " .\n";
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
    std::cerr << "Unrecognized vector variable " << var_name << " .\n";
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
  if (i_neuron>=n_neurons_) {
    std::cerr << "Neuron index must be lower then n. of neurons\n";
    exit(-1);
  }
  else if (i_neuron<0) {
    std::cerr << "Neuron index must be >= 0\n";
    exit(-1);
  }
  return 0;
}

int BaseNeuron::CheckReceptorIdx(int i_receptor)
{
  if (i_receptor>=n_receptors_) {
    std::cerr << "Receptor index must be lower then n. of receptors\n";
    exit(-1);
  }
  else if (i_receptor<0) {
    std::cerr << "Receptor index must be >= 0\n";
    exit(-1);
  }
  return 0;
}

float *BaseNeuron::GetVarPt(std::string var_name, int i_neuron, int i_receptor)
{
  CheckNeuronIdx(i_neuron);
  CheckReceptorIdx(i_receptor);
    
  if (IsScalVar(var_name)) {
    int i_var =  GetScalVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + i_var;
  }
  else if (IsVectVar(var_name)) {
    int i_vvar =  GetVectVarIdx(var_name);
    return GetVarArr() + i_neuron*n_var_ + n_scal_var_
      + i_receptor*n_vect_var_ + i_vvar;
  }
  else {
    std::cerr << "Unrecognized variable " << var_name << " .\n";
    exit(-1);
  }
  return NULL;
}

float *BaseNeuron::GetParamPt(std::string param_name, int i_neuron,
			      int i_receptor)
{
  CheckNeuronIdx(i_neuron);
  CheckReceptorIdx(i_receptor);
    
  if (IsScalParam(param_name)) {
    int i_param =  GetScalParamIdx(param_name);
    return GetParamsArr() + i_neuron*n_params_ + i_param;
  }
  else if (IsVectParam(param_name)) {
    int i_vparam =  GetVectParamIdx(param_name);
    return GetParamsArr() + i_neuron*n_params_ + n_scal_params_
      + i_receptor*n_vect_params_ + i_vparam;
  }
  else {
    std::cerr << "Unrecognized parameter " << param_name << " .\n";
    exit(-1);
  }
  return NULL;
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

