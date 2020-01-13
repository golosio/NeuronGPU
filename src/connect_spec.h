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

#ifndef CONNECTSPECH
#define CONNECTSPECH

#include <iostream>

class NeuralGPU;

struct RemoteNode
{
  int i_host_;
  int i_node_;
};

struct RemoteNodePt
{
  int i_host_;
  int *i_node_;
};

enum ConnectionRules
  {
   ONE_TO_ONE=0, ALL_TO_ALL, FIXED_TOTAL_NUMBER, FIXED_INDEGREE,
   FIXED_OUTDEGREE, N_CONN_RULE
  };

const std::string conn_rule_name[N_CONN_RULE] =
  {
   "one_to_one", "all_to_all", "fixed_total_number", "fixed_indegree",
   "fixed_outdegree"
};

class ConnSpec
{
  int rule_;
  int total_num_;
  int indegree_;
  int outdegree_;
public:
  ConnSpec();
  ConnSpec(int rule, int degree=0);
  int Init();
  int Init(int rule, int degree=0);
  int SetParam(std::string param_name, int value);
  int GetParam(std::string param_name);
  static bool IsParam(std::string param_name);

  friend class NeuralGPU;
};

enum SynapseTypes
  {
   STANDARD_SYNAPSE=0, STDP, N_SYNAPSE_TYPE
  };

const std::string synapse_type_name[N_SYNAPSE_TYPE] =
  {
   "standard_synapse", "stdp"
  };

class SynSpec
{
  unsigned char synapse_type_;
  unsigned char port_;
 public:
  int weight_distr_;
  float *weight_array_;
  float weight_;
  int delay_distr_;
  float *delay_array_;
  float delay_;
 public:
  SynSpec();
  SynSpec(float weight, float delay);
  SynSpec(int syn_type, float weight, float delay, int port=0);
  int Init();
  int Init(float weight, float delay);
  int Init(int syn_type, float weight, float delay, int port=0);
  int SetParam(std::string param_name, int value);
  int SetParam(std::string param_name, float value);
  int SetParam(std::string param_name, float *array_pt);
  float GetParam(std::string param_name);
  static bool IsIntParam(std::string param_name);
  static bool IsFloatParam(std::string param_name);
  static bool IsFloatPtParam(std::string param_name);

  friend class NeuralGPU;
};

#endif
