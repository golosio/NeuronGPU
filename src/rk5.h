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

#ifndef RK5H
#define RK5H

#include "cuda_error.h"
#include "rk5_const.h"
#include "rk5_interface.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MAXNVAR 6 //20
#define MAXNPARAM 21 //40

__global__ void SetFloatArray(float *arr, int n_elem, int step, float val);

template<class DataStruct>
__global__
void ArrayInit(int array_size, int n_var, int n_param, float *x_arr,
	       float *h_arr, float *y_arr, float *par_arr, float x_min,
	       float h, DataStruct data_struct)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    NodeInit(n_var, n_param, x_min, &y_arr[array_idx*n_var],
	     &par_arr[array_idx*n_param], data_struct);
    x_arr[array_idx] = x_min;
    h_arr[array_idx] = h;
  }
}

template<class DataStruct>
__global__
void ArrayCalibrate(int array_size, int n_var, int n_param, float *x_arr,
		    float *h_arr, float *y_arr, float *par_arr, float x_min,
		    float h, DataStruct data_struct)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<array_size) {
    NodeCalibrate(n_var, n_param, x_min, &y_arr[array_idx*n_var],
		  &par_arr[array_idx*n_param], data_struct);
    x_arr[array_idx] = x_min;
    h_arr[array_idx] = h;
  }
}

template<class DataStruct>
__device__
void RK5Step(int i_array, int i_var, float &x, float *y, float &h,
	     float h_min, float h_max, float *y_new,
	     float *k1, float *k2, float *k3, float *k4,
	     float *k5, float *k6, float *err_arr, int n_var, int n_param,
	     float *param,
	     DataStruct data_struct)
{
  for(;;) {
    if (h > h_max) h = h_max;

    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x, y, k1, n_var, n_param, param, data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*a21*k1[i_var];
    //printf("okk0 %d %f\n", i_var, y_new[i_var]);

    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x+c2*h, y_new, k2, n_var, n_param, param,
			      data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*(a31*k1[i_var] + a32*k2[i_var]);
    //printf("okk1 %d %f\n", i_var, y_new[i_var]);
    
    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x+c3*h, y_new, k3, n_var, n_param, param,
			      data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*(a41*k1[i_var] + a42*k2[i_var] + a43*k3[i_var]);
    //printf("okk2 %d %f\n", i_var, y_new[i_var]);
    
    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x+c4*h, y_new, k4, n_var, n_param, param,
			      data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*(a51*k1[i_var] + a52*k2[i_var] + a53*k3[i_var]
				 + a54*k4[i_var]);
    //printf("okk3 %d %f\n", i_var, y_new[i_var]);
    
    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x+c5*h, y_new, k5, n_var, n_param, param,
			      data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*(a61*k1[i_var] + a62*k2[i_var] + a63*k3[i_var]
				 + a64*k4[i_var] + a65*k5[i_var]);
    //printf("okk4 %d %f\n", i_var, y_new[i_var]);
    
    float x1 = x + h;
    
    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x1, y_new, k6, n_var, n_param, param,
			      data_struct);
    }
    __syncthreads();
    y_new[i_var] = y[i_var] + h*(a71*k1[i_var] + a73*k3[i_var] + a74*k4[i_var]
				 + a75*k5[i_var] + a76*k6[i_var]);
    //printf("okk5 %d %f\n", i_var, y_new[i_var]);
    
    __syncthreads();
    if (i_var==0) {
      Derivatives<DataStruct>(x1, y_new, k2, n_var, n_param, param,
			      data_struct); // k2 replaces k7
    }
    __syncthreads();
    float val = h*(e1*k1[i_var] + e3*k3[i_var] + e4*k4[i_var] + e5*k5[i_var]
		   + e6*k6[i_var] + e7*k2[i_var])
      / (abs_tol + rel_tol*MAX(fabs(y[i_var]), fabs(y_new[i_var])));
    //printf("okk6 %d %f\n", i_var, val);
    
    err_arr[i_var] = val*val;
    __syncthreads();
    float x_new = x + h;
    if (i_var==0) {
      float err = 0.0;
      for (int i=0; i<n_var; i++) {
	err += err_arr[i];
      }
      err = sqrt(err/n_var);
      //printf("okk7 %f\n", err);
      if (err<min_err) err = min_err;
      if (err>max_err) err = max_err;
      err_arr[0] = err;
    }
    __syncthreads();
    float err = err_arr[0];
    bool rejected=false;    
    float fact=coeff*pow(err,-alpha);
    //printf("okk8 %f\n", fact);
    if (fact>1.0) fact=1.0;
    h *= fact;
    
    //printf("okk9 %f\n", h);
    if (h <= h_min) {
      h = h_min;
      rejected = false;
    }
    else if (err <= 1.0) rejected = false; 
    else rejected = true;
    if (!rejected) {
      x = x_new;
      break;
    }
  }
  //printf("okk10 %d\n", i_var);
  y[i_var] = y_new[i_var];
  __syncthreads();
}

template<class DataStruct>
__device__
void RK5Update(int i_array, int i_var, float &x, float *y, float x1,
	       float &h, float h_min, float *y_new,
	       float *k1, float *k2, float *k3, float *k4, float *k5,
	       float *k6, float *err_arr, int n_var, int n_param, float *param,
	       DataStruct data_struct)
{
  bool end_time_step=false;
  while(!end_time_step) {
    float hmax=x1-x;
    RK5Step<DataStruct>(i_array, i_var, x, y, h, h_min, hmax, y_new,
			k1, k2, k3, k4, k5, k6, err_arr,
			n_var, n_param, param, data_struct);
    end_time_step = (x >= x1-h_min);
    if (i_var==0) {
      ExternalUpdate<DataStruct>(x, y, n_var, n_param, param, end_time_step,
				 data_struct);
    }
  }
}

template<class DataStruct>
__global__
void ArrayUpdate(int array_size, float *x_arr, float *h_arr, float *y_arr,
		 float *par_arr, float x1, float h_min, int n_var, int n_param,
		 DataStruct data_struct)
{
  //extern __shared__ shared_data[];
  __shared__ float shared_data[48*1024/4];
  //__shared__ float shared_data[30*256]; //24*1024/4];

  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  int i_var = threadIdx.y; // + blockIdx.y * blockDim.y;
  if (i_array>=array_size || i_var>=n_var) return;

  int thread_x = threadIdx.x;
  float *param = shared_data + thread_x*n_param; //[MAXNPARAM];
  float *x_pt = shared_data + blockDim.x*n_param + thread_x;
  float *h_pt = x_pt + blockDim.x;
  float *shared_var = shared_data + blockDim.x*(n_param + 2); 
  float *y = shared_var + thread_x*n_var; //[MAXNVAR];
  float *y_new = y + blockDim.x*n_var; //[MAXNVAR];
  float *k1 = y_new + blockDim.x*n_var; //[MAXNVAR];
  float *k2 = k1 + blockDim.x*n_var; //[MAXNVAR];
  float *k3 = k2 + blockDim.x*n_var; //[MAXNVAR];
  float *k4 = k3 + blockDim.x*n_var; //[MAXNVAR];
  float *k5 = k4 + blockDim.x*n_var; //[MAXNVAR];
  float *k6 = k5 + blockDim.x*n_var; //[MAXNVAR];
  float *err_arr = k6 + blockDim.x*n_var; //[MAXNVAR];
  //printf("ok0 %d %d\n", i_array, i_var);
  if (i_var==0) {
    *x_pt = x_arr[i_array];
    *h_pt = h_arr[i_array];
    //printf("ok1 %f %f\n", *x_pt, *h_pt);
  }
  __syncthreads();
  float x = *x_pt;
  float h = *h_pt;

  for(int j=0; j<=(n_param-1)/n_var; j++) {
    int i_param = i_var + j*n_var;
    if (i_param<n_param) {
      param[i_param] = par_arr[i_array*n_param + i_param];
      //printf("ok2 %d %f\n", i_param, param[i_param]);
    }
  }

  y[i_var] = y_arr[i_array*n_var + i_var];
  //printf("ok3 %d %f\n", i_var, y[i_var]);
  __syncthreads();
  //printf("ok4\n");
  RK5Update<DataStruct>(i_array, i_var, x, y, x1, h, h_min, y_new,
			k1, k2, k3, k4, k5, k6, err_arr,
			n_var, n_param, param, data_struct);

  //printf("ok5 %f %f\n", x, h);
  if (i_var==0) {
    x_arr[i_array] = x;
    h_arr[i_array] = h;
  }
  y_arr[i_array*n_var + i_var] = y[i_var];
  //printf("ok6 %d %f\n", i_var, y[i_var]);
}

template<class DataStruct>
class RungeKutta5
{
  int array_size_;
  int n_var_;
  int n_param_;
    
  float *d_XArr;
  float *d_HArr;
  float *d_YArr;
  float *d_ParamArr;

  public:

  ~RungeKutta5();
 
  float *GetXArr() {return d_XArr;}
  float *GetHArr() {return d_HArr;}
  float *GetYArr() {return d_YArr;}
  float *GetParamArr() {return d_ParamArr;}
  int Init(int array_size, int n_var, int n_param, float x_min, float h,
	   DataStruct data_struct);
  int Calibrate(float x_min, float h, DataStruct data_struct);

  int Free();

  int GetX(int i_array, int n_elem, float *x);
  int GetY(int i_var, int i_array, int n_elem, float *y);
  int SetParam(int i_param, int i_array, int n_param, int n_elem, float val);
  int SetVectParam(int i_param, int i_array, int n_param, int n_elem,
		    float *param, int vect_size);
  int Update(float x1, float h_min, int n_var, int n_param,
	     DataStruct data_struct);

};


template<class DataStruct>
  int RungeKutta5<DataStruct>::Update(float x1, float h_min,
				      int n_var, int n_param,
				      DataStruct data_struct)
{
  //ArrayUpdate<DataStruct><<<(array_size_+1023)/1024, 1024>>>
  //ArrayUpdate<DataStruct><<<(array_size_+127)/128, 128>>>
  //ArrayUpdate<DataStruct><<<(array_size_+255)/256, 256>>>
  dim3 dimBlock(128, n_var);
  dim3 dimGrid((array_size_+127)/128, 1);
  ArrayUpdate<DataStruct><<<dimGrid, dimBlock>>>
    (array_size_, d_XArr, d_HArr, d_YArr, d_ParamArr, x1, h_min, n_var,
     n_param, data_struct);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template<class DataStruct>
RungeKutta5<DataStruct>::~RungeKutta5()
{
  Free();
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Free()
{
  cudaFree(d_XArr);
  cudaFree(d_HArr);
  cudaFree(d_YArr);
  cudaFree(d_ParamArr);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Init(int array_size, int n_var, int n_param,
				  float x_min, float h,
				  DataStruct data_struct)
{
  array_size_ = array_size;
  n_var_ = n_var;
  n_param_ = n_param; 

  gpuErrchk(cudaMalloc(&d_XArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_HArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_YArr, array_size_*n_var_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_ParamArr, array_size_*n_param_*sizeof(float)));

  ArrayInit<DataStruct><<<(array_size+1023)/1024, 1024>>>
    (array_size_, n_var, n_param, d_XArr, d_HArr, d_YArr, d_ParamArr,
     x_min, h, data_struct);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Calibrate(float x_min, float h,
				       DataStruct data_struct)
{
  ArrayCalibrate<DataStruct><<<(array_size_+1023)/1024, 1024>>>
    (array_size_, n_var_, n_param_, d_XArr, d_HArr, d_YArr, d_ParamArr,
     x_min, h, data_struct);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetX(int i_array, int n_elem, float *x)
{
  cudaMemcpy(x, &d_XArr[i_array], n_elem*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetY(int i_var, int i_array, int n_elem, float *y)
{
  cudaMemcpy(y, &d_YArr[i_array*n_var_ + i_var], n_elem*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::SetParam(int i_param, int i_array, int n_param,
			   int n_elem, float val)
{
  SetFloatArray<<<(n_elem+1023)/1024, 1024>>>
    (&d_ParamArr[i_array*n_param_ + i_param], n_elem, n_param, val);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

#endif
