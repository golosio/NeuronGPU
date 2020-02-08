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
#include "neuralgpu.h"
#include "syn_model.h"
#include "test_syn_model.h"

int SynModel::GetNParam()
{
  return n_param_;
}

std::vector<std::string> SynModel::GetParamNames()
{
  std::vector<std::string> param_name_vect;
  for (int i=0; i<n_param_; i++) {
    param_name_vect.push_back(param_name_[i]);
  }
  
  return param_name_vect;
}

bool SynModel::IsParam(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_param_; i_param++) {
    if (param_name == param_name_[i_param]) return true;
  }
  return false;
}

int SynModel::GetParamIdx(std::string param_name)
{
  int i_param;
  for (i_param=0; i_param<n_param_; i_param++) {
    if (param_name == param_name_[i_param]) break;
  }
  if (i_param == n_param_) {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
  
  return i_param;
}

float SynModel::GetParam(std::string param_name)
{
  if (!IsParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized synapse parameter ")
			 + param_name);
  }
  int i_param =  GetParamIdx(param_name);
  float *d_param_pt = d_param_arr_ + i_param;
  float param_val;
  gpuErrchk(cudaMemcpy(&param_val, d_param_pt, sizeof(float),
		       cudaMemcpyDeviceToHost));
  return param_val;
}


int SynModel::SetParam(std::string param_name, float val)
{
  if (!IsParam(param_name)) {
    throw ngpu_exception(std::string("Unrecognized synapse parameter ")
			 + param_name);
  }
  int i_param =  GetParamIdx(param_name);
  float *d_param_pt = d_param_arr_ + i_param;
  gpuErrchk(cudaMemcpy(d_param_pt, &val, sizeof(float),
		       cudaMemcpyHostToDevice));
  return 0;
}

  
int NeuralGPU::CreateSynGroup(std::string model_name)
{
  CheckUncalibrated("Nodes cannot be created after calibration");
  if (model_name == syn_model_name[i_test_syn_model]) {
    TestSynModel *test_syn_model_group = new TestSynModel;
    syn_group_vect_.push_back(test_syn_model_group);
  }
  else {
    throw ngpu_exception(std::string("Unknown synapse model name: ")
			 + model_name);
  }
  return (syn_group_vect_.size() - 1);
}

int NeuralGPU::GetSynGroupNParam(int syn_group)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->GetNParam();
}

std::vector<std::string> NeuralGPU::GetSynGroupParamNames(int syn_group)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->GetParamNames();
}

bool NeuralGPU::IsSynGroupParam(int syn_group, std::string param_name)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->IsParam(param_name);
}

int NeuralGPU::GetSynGroupParamIdx(int syn_group, std::string param_name)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->GetParamIdx(param_name);
}

float NeuralGPU::GetSynGroupParam(int syn_group, std::string param_name)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->GetParam(param_name);
}

int NeuralGPU::SetSynGroupParam(int syn_group, std::string param_name,
				float val)
{
  if (syn_group<0 || syn_group>(int)syn_group_vect_.size()) {
    throw ngpu_exception("Unrecognized synapse group");
  }

  return syn_group_vect_[syn_group]->SetParam(param_name, val);
}
