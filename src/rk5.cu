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

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "rk5.h"

using namespace std;

__device__ int ARRAY_SIZE;

__device__ float *XArr;
__device__ float *HArr;
__device__ float *YArr;
__device__ float *ParamsArr;

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


__global__
void ArrayDef(int array_size, float *x_arr, float *h_arr, float *y_arr,
	      float *par_arr)
{
  ARRAY_SIZE = array_size;
  XArr = x_arr;
  HArr = h_arr;
  YArr = y_arr;
  ParamsArr = par_arr;
}
  
__global__
void ArrayInit(int n_var, int n_params, float x_min, float h)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<ARRAY_SIZE) {
    VarInit(n_var, n_params, x_min, &YArr[array_idx*n_var],
	    &ParamsArr[array_idx*n_params]);
    XArr[array_idx] = x_min;
    HArr[array_idx] = h;
  }
}

__global__
void ArrayCalibrate(int n_var, int n_params, float x_min, float h)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<ARRAY_SIZE) {
    VarCalibrate(n_var, n_params, x_min, &YArr[array_idx*n_var],
		 &ParamsArr[array_idx*n_params]);
    XArr[array_idx] = x_min;
    HArr[array_idx] = h;
  }
}

RungeKutta5::~RungeKutta5()
{
  Free();
}

int RungeKutta5::Free()
{
  cudaFree(d_XArr);
  cudaFree(d_HArr);
  cudaFree(d_YArr);
  cudaFree(d_ParamsArr);

  return 0;
}

int RungeKutta5::Init(int array_size, int n_var, int n_params, float x_min,
				     float h)
{
  array_size_ = array_size;
  n_var_ = n_var;
  n_params_ = n_params; 

  gpuErrchk(cudaMalloc(&d_XArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_HArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_YArr, array_size_*n_var_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_ParamsArr, array_size_*n_params_*sizeof(float)));

  ArrayDef<<<1, 1>>>(array_size_, d_XArr, d_HArr, d_YArr, d_ParamsArr);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  //ArrayAlloc();

  ArrayInit<<<(array_size+1023)/1024, 1024>>>(n_var, n_params,
	     x_min, h);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int RungeKutta5::Calibrate(float x_min, float h)
{
  ArrayCalibrate<<<(array_size_+1023)/1024, 1024>>>(n_var_, n_params_,
	     x_min, h);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int RungeKutta5::GetX(int i_array, int n_elems, float *x)
{
  cudaMemcpy(x, &d_XArr[i_array], n_elems*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

int RungeKutta5::GetY(int i_var, int i_array, int n_elems, float *y)
{
  cudaMemcpy(y, &d_YArr[i_array*n_var_ + i_var], n_elems*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}


__global__ void SetFloatArray(float *arr, int n_elems, int step, float val)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<n_elems) {
    arr[array_idx*step] = val;
  }
}

int RungeKutta5::SetParams(int i_param, int i_array, int n_params,
			   int n_elems, float val)
{
  // TMP
  //printf("rk5::SetParams %d %d %d %d %f\n",
  //	 i_param, i_array, n_params_, n_elems, val);
  //

  SetFloatArray<<<(n_elems+1023)/1024, 1024>>>
    (&d_ParamsArr[i_array*n_params_ + i_param], n_elems, n_params, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

int RungeKutta5::SetVectParams(int i_param, int i_array, int n_params,
			       int n_elems, float *params, int vect_size)
{
  // TMP
  //printf("rk5::SetVectParams %d %d %d %d\n",
  //	 i_param, i_array, n_params_, n_elems);
  //for (int i=0; i<vect_size; i++) {
  //  printf("rk5::SetVectParams vect %d %d %f\n",
  //	   i_param, i, params[i]);
  //}
  //

  for (int i=0; i<vect_size; i++) {
    SetFloatArray<<<(n_elems+1023)/1024, 1024>>>
      (&d_ParamsArr[i_array*n_params_ + N0_PARAMS + i_param + 4*i], n_elems,
       n_params, params[i]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  return 0;
}
