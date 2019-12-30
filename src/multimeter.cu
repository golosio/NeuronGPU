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

#include "multimeter.h"
#include <vector>

using namespace std;

Record::Record(BaseNeuron *neuron, std::string file_name, std::string var_name,
	       int *i_neurons, int n_neurons):
  neuron_(neuron), file_name_(file_name), var_name_(var_name)
{
  for (int i=0; i<n_neurons; i++) {
    i_neurons_.push_back(i_neurons[i]);
  }
  i_var_=neuron_->GetScalVarIdx(var_name_);
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

int Record::WriteRecord()
{
  float x, y;
  neuron_->GetX(i_neurons_[0], 1, &x);
  fprintf(fp_,"%f", x);
  for (unsigned int i=0; i<i_neurons_.size(); i++) {
    neuron_->GetY(i_var_, i_neurons_[i], 1, &y);
    fprintf(fp_,"\t%f", y);
  }
  fprintf(fp_,"\n");

  return 0;
}

int Multimeter::CreateRecord(BaseNeuron *neuron, std::string file_name,
			     std::string var_name, int *i_neurons,
			     int n_neurons)
{
  Record record(neuron, file_name, var_name, i_neurons, n_neurons);
  record_array_.push_back(record);

  return 0;
}

int Multimeter::OpenFiles()
{
  for (unsigned int i=0; i<record_array_.size(); i++) {
    record_array_[i].OpenFile();
  }
  
  return 0;
}

int Multimeter::CloseFiles()
{  
  for (unsigned int i=0; i<record_array_.size(); i++) {
    record_array_[i].CloseFile();
  }
  
  return 0;
}

int Multimeter::WriteRecords()
{  
  for (unsigned int i=0; i<record_array_.size(); i++) {
    record_array_[i].WriteRecord();
  }
  
  return 0;
}
