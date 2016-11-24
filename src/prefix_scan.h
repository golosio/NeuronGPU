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

#ifndef PREFIX_SCAN_H
#define PREFIX_SCAN_H

class PrefixScan
{
 public:
  static const uint AllocSize;

  /*
  uint *d_Input;

  uint *d_Output;

  uint *h_Input;

  uint *h_OutputCPU;

  uint *h_OutputGPU;
  */
  
  int Init();

  int Scan(uint *d_Output, uint *d_Input, uint n);

  int Free();
};

#endif
