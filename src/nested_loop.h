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

#ifndef NESTED_LOOP_H
#define  NESTED_LOOP_H

#define WITH_CUMUL_SUM
#ifdef WITH_CUMUL_SUM
#include "prefix_scan.h"
#endif

namespace NestedLoop
{
  extern int Nx_max_;
  extern int block_dim_x_;
  extern int block_dim_y_;
  extern int frame_area_;
  extern float x_lim_;
  extern const int Ny_arr_size_;
  extern int Ny_th_arr_[];
    
  int Init(int Nx_max);
  int Init();
  int Run(int Nx, int *d_Ny);
  int SimpleNestedLoop(int Nx, int *d_Ny);
  int SimpleNestedLoop(int Nx, int *d_Ny, int max_Ny);
  int ParallelInnerNestedLoop(int Nx, int *d_Ny);
  int ParallelOuterNestedLoop(int Nx, int *d_Ny);
  int Frame1DNestedLoop(int Nx, int *d_Ny);
  int Frame2DNestedLoop(int Nx, int *d_Ny);
  int Smart1DNestedLoop(int Nx, int *d_Ny);
  int Smart2DNestedLoop(int Nx, int *d_Ny);
  #ifdef WITH_CUMUL_SUM
  //extern PrefixScan prefix_scan_;
  int CumulSumNestedLoop(int Nx, int *d_Ny);  
  #endif

  int Free();
}

#endif
