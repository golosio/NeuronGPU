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
#include <cmath>
#include <iostream>
#include "fast_iaf_psc_exp.h"
#include "spike_buffer.h"

using namespace fast_iaf_psc_exp_ns;

extern __constant__ float NeuronGPUTimeResolution;

#define I_syn var[i_I_syn]
#define V_m_rel var[i_V_m_rel]
#define refractory_step var[i_refractory_step]

#define I_e param[i_I_e]
#define tau_m 10.0
#define C_m 250
#define E_L -65.0
#define Theta_rel 15.0
#define V_reset_rel 0.0
#define tau_syn param[i_tau_ex]
#define t_ref 2.0
#define n_refractory_steps 20

#define exp_tau_m 0.9900498337491681
//#define Rm 40.0
#define Rm 0.04

#define exp_tau_syn 0.8187307530779818

__global__ void fast_iaf_psc_exp_Update(int n_node, int i_node_0,
					float *var_arr, float *param_arr,
					int n_var, int n_param)
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_neuron<n_node) {
    float *var = var_arr + n_var*i_neuron;
    float *param = param_arr + n_param*i_neuron;
    
    if ( refractory_step > 0.0 ) {
      // neuron is absolute refractory
      refractory_step -= 1.0;
    }
    else { // neuron is not refractory, so evolve V
      double RI = (I_syn + I_e)* Rm;
      V_m_rel = exp_tau_m*V_m_rel + (1.0 - exp_tau_m)*RI;
    }
    // exponential decaying PSCs
    I_syn *= exp_tau_syn;
    
    if (V_m_rel >= Theta_rel ) { // threshold crossing
      PushSpike(i_node_0 + i_neuron, 1.0);
      V_m_rel = V_reset_rel;
      refractory_step = n_refractory_steps;
    }    
  }
}

fast_iaf_psc_exp::~fast_iaf_psc_exp()
{
  FreeVarArr();
  FreeParamArr();
}

int fast_iaf_psc_exp::Init(int i_node_0, int n_node, int /*n_port*/,
			   int i_group, unsigned long long *seed)
{
  BaseNeuron::Init(i_node_0, n_node, 1 /*n_port*/, i_group, seed);
  node_type_ = i_fast_iaf_psc_exp_model;

  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;
  
  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = fast_iaf_psc_exp_scal_var_name;
  scal_param_name_ = fast_iaf_psc_exp_scal_param_name;

  SetScalParam(0, n_node, "I_e", 0.0 );              // in pA

  SetScalVar(0, n_node, "I_syn", 0.0 );
  SetScalVar(0, n_node, "V_m_rel", 0.0 ); // in mV
  SetScalVar(0, n_node, "refractory_step", 0 );

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  gpuErrchk(cudaMalloc(&port_weight_arr_, sizeof(float)));
  gpuErrchk(cudaMemcpy(port_weight_arr_, &input_weight,
			 sizeof(float), cudaMemcpyHostToDevice));
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;
  
  // input spike signal is stored in I_syn
  port_input_arr_ = GetVarArr() + GetScalVarIdx("I_syn");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 0;

  return 0;
}

int fast_iaf_psc_exp::Update(int it, float t1)
{
  // std::cout << "fast_iaf_psc_exp neuron update\n";
  fast_iaf_psc_exp_Update<<<(n_node_+1023)/1024, 1024>>>
    (n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_);
  //gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int fast_iaf_psc_exp::Free()
{
  FreeVarArr();  
  FreeParamArr();
  
  return 0;
}

