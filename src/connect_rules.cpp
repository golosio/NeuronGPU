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
#include "connect.h"
#include "neuralgpu.h"
#include "connect_rules.h"

ConnSpec::ConnSpec()
{
  rule_ = ALL_TO_ALL;
  total_num_ = 0;
  indegree_ = 0;
  outdegree_ = 0;
}

ConnSpec::ConnSpec(int rule, int degree /*=0*/)
{
  ConnSpec();
  if (rule<0 || rule>N_CONN_RULE) {
    std::cerr << "Unknown connection rule\n";
    exit(0);
  }
  if ((rule==ALL_TO_ALL || rule==ONE_TO_ONE) && (degree != 0)) {
    std::cerr << "Connection rule " << conn_rule_name[rule]
	      << " does not have a degree\n";
  }
  rule_ = rule;
  if (rule==FIXED_TOTAL_NUMBER) {
    total_num_ = degree;
  }
  else if (rule==FIXED_INDEGREE) {
    indegree_ = degree;
  }
  else if (rule==FIXED_OUTDEGREE) {
    outdegree_ = degree;
  }
}

int ConnSpec::SetParam(std::string param_name, int value)
{
  if (param_name=="rule") {
    if (value<0 || value>N_CONN_RULE) {
      std::cerr << "Unknown connection rule\n";
      exit(0);
    }
    rule_ = value;
    return 0;
  }
  else if (param_name=="indegree") {
    if (value<0) {
      std::cerr << "Indegree must be >=0\n";
      exit(0);
    }
    indegree_ = value;
    return 0;
  }
  else if (param_name=="outdegree") {
    if (value<0) {
      std::cerr << "Outdegree must be >=0\n";
      exit(0);
    }
    outdegree_ = value;
    return 0;
  }
  else {
    std::cerr << "Unknown connection int parameter\n";
    exit(0);
  }
  return 0;
}
//int ConnSpec::GetParam(std::string param_name);

SynSpec::SynSpec()
{
  Init();
}


int SynSpec::Init()
{
  synapse_type_ = STANDARD_SYNAPSE;
  receptor_ = 0;
  weight_ = 0;
  delay_ = 0;
  weight_distr_ = 0;
  delay_distr_ = 0;
  weight_array_ = NULL;
  delay_array_ = NULL;

  return 0;
}


SynSpec::SynSpec(float weight, float delay)
{
  Init(weight, delay);
}

int SynSpec::Init(float weight, float delay)
{
  if (delay<0) {
    std::cerr << "Delay must be >=0\n";
    exit(0);
  }
  Init();
  weight_ = weight;
  delay_ = delay;

  return 0;
 }

SynSpec::SynSpec(int syn_type, float weight, float delay, int receptor /*=0*/)
{
  Init(syn_type, weight, delay, receptor);
}

int SynSpec::Init(int syn_type, float weight, float delay, int receptor /*=0*/)
{
  if (syn_type<0 || syn_type>N_SYNAPSE_TYPE) {
    std::cerr << "Unknown synapse type\n";
    exit(0);
  }
  if (receptor<0) {
    std::cerr << "Receptor index must be >=0\n";
    exit(0);
  }
  Init(weight, delay);
  synapse_type_ = syn_type;
  receptor_ = receptor;

  return 0;
 }

int SynSpec::SetParam(std::string param_name, int value)
{
  if (param_name=="synapse_type") {
    if (value<0 || value>N_SYNAPSE_TYPE) {
      std::cerr << "Unknown synapse type\n";
      exit(0);
    }
    synapse_type_ = value;
    return 0;
  }
  else if (param_name=="receptor") {
    if (value<0) {
      std::cerr << "Receptor index must be >=0\n";
      exit(0);
    }
    receptor_ = value;
    return 0;
  }
  else {
    std::cerr << "Unknown synapse int parameter\n";
    exit(0);
  }
  return 0;
}

int SynSpec::SetParam(std::string param_name, float value)
{
  if (param_name=="weight") {
    weight_ = value;
  }
  else if (param_name=="delay") {
    if (value<0) {
      std::cerr << "Delay must be >=0\n";
      exit(0);
    }
    delay_ = value;
  }
  else {
    std::cerr << "Unknown synapse float parameter\n";
    exit(0);
  }
  return 0;
}

 
int SynSpec::SetParam(std::string param_name, float *array_pt)
{
  if (param_name=="weight_array") {
    weight_array_ = array_pt;
  }
  else if (param_name=="delay_array") {
    delay_array_ = array_pt;
  }
  else {
    std::cerr << "Unknown synapse array parameter\n";
    exit(0);
  }
  
  return 0;
}
//float SynSpec::GetParam(std::string param_name);

template<>
int NeuralGPU::_SingleConnect<int>(int i_source0, int i_source, int i_target0,
				   int i_target, float weight, float delay,
				   int i_array, SynSpec &syn_spec)
{
  //return SingleConnect(i_source0 + i_source, i_target0 + i_target,
  // weight, delay, i_array, syn_spec);
  return net_connection_->Connect(i_source0 + i_source, i_target0 + i_target,
				  syn_spec.receptor_, weight, delay);
}

template<>
int NeuralGPU::_SingleConnect<int*>(int *i_source0, int i_source,
				    int *i_target0, int i_target,
				    float weight, float delay,
				    int i_array, SynSpec &syn_spec)
{
  //return SingleConnect(*(i_source0 + i_source), *(i_target0 + i_target),
  //		       weight, delay, i_array, syn_spec);
  return net_connection_->Connect(*(i_source0 + i_source),
				  *(i_target0 + i_target),
				  syn_spec.receptor_, weight, delay);
}

/*
template<>
int NeuralGPU::_SingleConnect<RemoteNeuron>(RemoteNeuron source, int i_source,
					    RemoteNeuron target, int i_target,
					    float weight, float delay,
					    int i_array, SynSpec &syn_spec)
{
  return RemoteSingleConnect(source.i_host_, source.i_neuron_ + i_source,
			     target.i_host_, target.i_neuron_ + i_target,
			     weight, delay, i_array, syn_spec);
}

template<>
int NeuralGPU::_SingleConnect<RemoteNeuronPt>(RemoteNeuronPt source,
					      int i_source,
					      RemoteNeuronPt target,
					      int i_target,
					      float weight, float delay,
					      int i_array, SynSpec &syn_spec)
{
  return RemoteSingleConnect(source.i_host_, *(source.i_neuron_ + i_source),
			     target.i_host_, *(target.i_neuron_ + i_target),
			     weight, delay, i_array, syn_spec);
}

*/
int NeuralGPU::Connect(int i_source, int n_source, int i_target, int n_target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _Connect<int>(i_source, n_source, i_target, n_target,
		       conn_spec, syn_spec);

}

int NeuralGPU::Connect(Nodes source, Nodes target,
		       ConnSpec &conn_spec, SynSpec &syn_spec)
{
  return _Connect<int>(source.i0, source.n, target.i0, target.n,
		       conn_spec, syn_spec);

}

