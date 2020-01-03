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

#include "multimeter.h"
#include "cuda_error.h"
#include <vector>

using namespace std;

const   std::string SpikeVarName = "spike";

Record::Record(std::vector<BaseNeuron*> neur_vect, std::string file_name,
	       std::vector<std::string> var_name_vect,
	       std::vector<int> i_neur_vect, std::vector<int> i_receptor_vect):
  neuron_vect_(neur_vect), file_name_(file_name),
  var_name_vect_(var_name_vect),
  i_neuron_vect_(i_neur_vect),
  i_receptor_vect_(i_receptor_vect)
{
  var_pt_vect_.clear();
  for (unsigned int i=0; i<var_name_vect.size(); i++) {
    if (var_name_vect[i]!=SpikeVarName) {
      float *var_pt = neur_vect[i]->GetVarPt(var_name_vect[i], i_neur_vect[i],
					     i_receptor_vect[i]);
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
  fprintf(fp_,"%f", t);
  for (unsigned int i=0; i<var_name_vect_.size(); i++) {
    if (var_name_vect_[i]!=SpikeVarName) {
      gpuErrchk(cudaMemcpy(&var, var_pt_vect_[i], sizeof(float),
			   cudaMemcpyDeviceToHost));
    }
    else {
      var = neuron_vect_[i]->GetSpikeActivity(i_neuron_vect_[i]);
    }
    fprintf(fp_,"\t%f", var);
  }
  fprintf(fp_,"\n");

  return 0;
}

int Multimeter::CreateRecord(std::vector<BaseNeuron*> neur_vect,
			     std::string file_name,
			     std::vector<std::string> var_name_vect,
			     std::vector<int> i_neur_vect,
			     std::vector<int> i_receptor_vect)
{
  Record record(neur_vect, file_name, var_name_vect, i_neur_vect,
		i_receptor_vect);
  record_vect_.push_back(record);

  return 0;
}

int Multimeter::OpenFiles()
{
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    record_vect_[i].OpenFile();
  }
  
  return 0;
}

int Multimeter::CloseFiles()
{  
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    record_vect_[i].CloseFile();
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
