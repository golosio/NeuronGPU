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
#include <vector>
#include "multimeter.h"
#include "cuda_error.h"

using namespace std;

const   std::string SpikeVarName = "spike";

Record::Record(std::vector<BaseNeuron*> neur_vect, std::string file_name,
	       std::vector<std::string> var_name_vect,
	       std::vector<int> i_neur_vect, std::vector<int> i_port_vect):
  neuron_vect_(neur_vect), file_name_(file_name),
  var_name_vect_(var_name_vect),
  i_neuron_vect_(i_neur_vect),
  i_port_vect_(i_port_vect)
{
  data_vect_flag_ = true;
  if (file_name=="") {
    out_file_flag_ = false;
  } else {
    out_file_flag_ = true;
  }
  var_pt_vect_.clear();
  for (unsigned int i=0; i<var_name_vect.size(); i++) {
    if (var_name_vect[i]!=SpikeVarName) {
      float *var_pt = neur_vect[i]->GetVarPt(var_name_vect[i], i_neur_vect[i],
					     i_port_vect[i]);
      var_pt_vect_.push_back(var_pt);
    }
  }
}

int Record::OpenFile()
{
  fp_=fopen(file_name_.c_str(), "w");
	   
  return 0;
}

int Record::CloseFile()
{
  fclose(fp_);
	   
  return 0;
}

int Record::WriteRecord(float t)
{
  float var;
  vector<float> vect;
  
  if (out_file_flag_) {
    fprintf(fp_,"%f", t);
  }
  if (data_vect_flag_) {
    vect.push_back(t);
  }
  for (unsigned int i=0; i<var_name_vect_.size(); i++) {
    if (var_name_vect_[i]!=SpikeVarName) {
      gpuErrchk(cudaMemcpy(&var, var_pt_vect_[i], sizeof(float),
			   cudaMemcpyDeviceToHost));
    }
    else {
      var = neuron_vect_[i]->GetSpikeActivity(i_neuron_vect_[i]);
    }
    if (out_file_flag_) {
      fprintf(fp_,"\t%f", var);
    }
    if (data_vect_flag_) {
      vect.push_back(var);
    }
  }
  if (out_file_flag_) {
    fprintf(fp_,"\n");
  }
  if (data_vect_flag_) {
    data_vect_.push_back(vect);
  }

  return 0;
}

int Multimeter::CreateRecord(std::vector<BaseNeuron*> neur_vect,
			     std::string file_name,
			     std::vector<std::string> var_name_vect,
			     std::vector<int> i_neur_vect,
			     std::vector<int> i_port_vect)
{
  Record record(neur_vect, file_name, var_name_vect, i_neur_vect,
		i_port_vect);
  record_vect_.push_back(record);

  return (record_vect_.size() - 1);
}

int Multimeter::OpenFiles()
{
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    if (record_vect_[i].out_file_flag_) {
      record_vect_[i].OpenFile();
    }
  }
  
  return 0;
}

int Multimeter::CloseFiles()
{  
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    if (record_vect_[i].out_file_flag_) {
      record_vect_[i].CloseFile();
    }
  }
  
  return 0;
}

int Multimeter::WriteRecords(float t)
{  
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    record_vect_[i].WriteRecord(t);
  }
  
  return 0;
}

std::vector<std::vector<float>> *Multimeter::GetRecordData(int i_record)
{
  if (i_record<0 || i_record>=(int)record_vect_.size()) {
    throw ngpu_exception("Record does not exist.");
  }
  
  return &record_vect_[i_record].data_vect_;
}
