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
#include <vector>

using namespace std;

Record::Record(std::vector<BaseNeuron*> neur_vect, std::string file_name,
	       std::vector<std::string> var_name_vect,
	       std::vector<int> i_neur_vect):
  neuron_vect_(neur_vect), file_name_(file_name),
  var_name_vect_(var_name_vect),
  i_neuron_vect_(i_neur_vect)
{
  i_var_vect_.clear();
  for (unsigned int i=0; i<var_name_vect.size(); i++) {
    int i_var = neur_vect[i]->GetScalVarIdx(var_name_vect[i]);
    i_var_vect_.push_back(i_var);
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

int Record::WriteRecord()
{
  float x, y;
  neuron_vect_[0]->GetX(i_neuron_vect_[0], 1, &x);
  fprintf(fp_,"%f", x);
  for (unsigned int i=0; i<i_neuron_vect_.size(); i++) {
    neuron_vect_[i]->GetY(i_var_vect_[i], i_neuron_vect_[i], 1, &y);
    fprintf(fp_,"\t%f", y);
  }
  fprintf(fp_,"\n");

  return 0;
}

int Multimeter::CreateRecord(std::vector<BaseNeuron*> neur_vect,
			     std::string file_name,
			     std::vector<std::string> var_name_vect,
			     std::vector<int> i_neur_vect)
{
  Record record(neur_vect, file_name, var_name_vect, i_neur_vect);
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

int Multimeter::WriteRecords()
{  
  for (unsigned int i=0; i<record_vect_.size(); i++) {
    record_vect_[i].WriteRecord();
  }
  
  return 0;
}
