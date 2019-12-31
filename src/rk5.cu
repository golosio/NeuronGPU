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
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "rk5.h"

__constant__ float c2 = 0.2;
__constant__ float c3 = 0.3;
__constant__ float c4 = 0.8;
__constant__ float c5 = 8.0/9.0;
__constant__ float a21 = 0.2;
__constant__ float a31 = 3.0/40.0;
__constant__ float a32 = 9.0/40.0;
__constant__ float a41 = 44.0/45.0;
__constant__ float a42 = -56.0/15.0;
__constant__ float a43 = 32.0/9.0;
__constant__ float a51 = 19372.0/6561.0;
__constant__ float a52 = -25360.0/2187.0;
__constant__ float a53 = 64448.0/6561.0;
__constant__ float a54 = -212.0/729.0;
__constant__ float a61 = 9017.0/3168.0;
__constant__ float a62 = -355.0/33.0;
__constant__ float a63 = 46732.0/5247.0;
__constant__ float a64 = 49.0/176.0;
__constant__ float a65 = -5103.0/18656.0;
__constant__ float a71 = 35.0/384.0;
__constant__ float a73 = 500.0/1113.0;
__constant__ float a74 = 125.0/192.0;
__constant__ float a75 = -2187.0/6784.0;
__constant__ float a76 = 11.0/84.0;
__constant__ float e1 = 71.0/57600.0;
__constant__ float e3 = -71.0/16695.0;
__constant__ float e4 = 71.0/1920.0;
__constant__ float e5 = -17253.0/339200.0;
__constant__ float e6 = 22.0/525.0;
__constant__ float e7 = -1.0/40.0;

__constant__ float abs_tol = 1.0e-8;
__constant__ float rel_tol = 1.0e-8;

__constant__ float min_err = 5.0e-6;
__constant__ float max_err = 2000.0;
__constant__ float coeff = 0.9;
__constant__ float alpha = 0.2;


__global__ void SetFloatArray(float *arr, int n_elems, int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elems) {
    arr[array_idx*step] = val;
  }
}

