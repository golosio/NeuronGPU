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

#ifndef MULTIMETERH
#define MULTIMETERH
#include <stdio.h>
#include <string>
#include <vector>
#include "base_neuron.h"

class Record
{
 public:
  bool data_vect_flag_;
  bool out_file_flag_;
  std::vector<std::vector<float>> data_vect_;
  std::vector<BaseNeuron*> neuron_vect_;
  std::string file_name_;
  std::vector<std::string> var_name_vect_;
  std::vector<int> i_neuron_vect_;
  std::vector<int> i_receptor_vect_;
  std::vector<float*> var_pt_vect_;
  FILE *fp_;

  Record(std::vector<BaseNeuron*> neur_vect, std::string file_name,
	 std::vector<std::string> var_name_vect,
	 std::vector<int> i_neur_vect, std::vector<int> i_receptor_vect);

  int OpenFile();
  
  int CloseFile();
  
  int WriteRecord(float t);

};
  
class Multimeter
{
 public:
  std::vector<Record> record_vect_;

  int CreateRecord(std::vector<BaseNeuron*> neur_vect,
		   std::string file_name,
		   std::vector<std::string> var_name_vect,
		   std::vector<int> i_neur_vect,
		   std::vector<int> i_receptor_vect);
  int OpenFiles();

  int CloseFiles();

  int WriteRecords(float t);

  std::vector<std::vector<float>> *GetRecordData(int i_record);
	     
};

#endif
