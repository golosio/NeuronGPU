/*
Copyright (C) 2016 Bruno Golosio
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
#include "aeif.h"

class Record
{
 public:
  AEIF *aeif_;
  std::string file_name_;
  std::string var_name_;
  std::vector<int> i_neurons_;
  int i_var_;
  FILE *fp_;

  Record(AEIF *aeif, std::string file_name, std::string var_name,
	 int *i_neurons, int n_neurons);

  int OpenFile();

  int CloseFile();

  int WriteRecord();

};
  
class Multimeter
{
 public:
  std::vector<Record> record_array_;

  int CreateRecord(AEIF *aeif, std::string file_name, std::string var_name,
		   int *i_neurons, int n_neurons);
  int OpenFiles();

  int CloseFiles();

  int WriteRecords();
	     
};

#endif
