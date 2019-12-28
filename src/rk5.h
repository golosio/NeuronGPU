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

#ifndef RK5H
#define RK5H

#include "cuda_error.h"
//extern __device__ int ARRAY_SIZE;
#include "rk5_const.h"
#include "derivatives.h"
#include "poisson.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

//extern __device__ float *XArr;
//extern __device__ float *HArr;
//extern __device__ float *YArr;
//extern __device__ float *ParamsArr;

__global__ void SetFloatArray(float *arr, int n_elems, int step, float val);

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void RK5Update(float &x, float *y, float x1, float &h, float h_min,
	       float *params, DataStruct data_struct);

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void RK5Step(float &x, float *y, float &h, float h_min, float h_max,
	     float *params, DataStruct data_struct);

template<int NVAR, int NPARAMS, class DataStruct>
__device__
  void ExternalUpdate(float x, float *y, float *params, bool end_time_step,
		      DataStruct data_struct);

//__global__
//void ArrayDef(int array_size, float *x_arr, float *h_arr, float *y_arr,
//	      float *par_arr);

__global__
void ArrayInit(int array_size, int n_var, int n_params, float *x_arr,
	       float *h_arr, float *y_arr, float *par_arr, float x_min,
	       float h);

__global__
void ArrayCalibrate(int array_size, int n_var, int n_params, float *x_arr,
		    float *h_arr, float *y_arr, float *par_arr, float x_min,
		    float h);

__device__
void VarInit(int array_size, int n_var, int n_params, float x, float *y,
	     float *params);

__device__
void VarCalibrate(int array_size, int n_var, int n_params, float x, float *y,
		  float *params);

template<int NVAR, int NPARAMS, class DataStruct>
__global__
void ArrayUpdate(int array_size, float *x_arr, float *h_arr, float *y_arr,
		 float *par_arr, float x1, float h_min, DataStruct data_struct);
template<int NVAR, int NPARAMS, class DataStruct>
__device__
  void Derivatives(float x, float *y, float *dydx, float *params,
		   DataStruct data_struct);

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void RK5Step(float &x, float *y, float &h, float h_min, float h_max,
	     float *params, DataStruct data_struct)
{
  float err;
  float y_new[NVAR];

  for(;;) {
    if (h > h_max) h = h_max;

    float k1[NVAR];
    float k2[NVAR];
    float k3[NVAR];
    float k4[NVAR];
    float k5[NVAR];
    float k6[NVAR];

    Derivatives<NVAR, NPARAMS, DataStruct>(x, y, k1, params, data_struct);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*a21*k1[i];
    }
    Derivatives<NVAR, NPARAMS, DataStruct>(x+c2*h, y_new, k2, params,
					   data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a31*k1[i] + a32*k2[i]);
    }
    Derivatives<NVAR, NPARAMS, DataStruct>(x+c3*h, y_new, k3, params,
					   data_struct);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3[i]);
    }
    Derivatives<NVAR, NPARAMS, DataStruct>(x+c4*h, y_new, k4, params,
					   data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
    }
    Derivatives<NVAR, NPARAMS, DataStruct>(x+c5*h, y_new, k5, params,
					   data_struct);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i]
			  + a65*k5[i]);
    }
    float x1 = x + h;
    Derivatives<NVAR, NPARAMS, DataStruct>(x1, y_new, k6, params, data_struct);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a71*k1[i] + a73*k3[i] + a74*k4[i] + a75*k5[i]
			  + a76*k6[i]);
    }
    Derivatives<NVAR, NPARAMS, DataStruct>(x1, y_new, k2, params,
					   data_struct); // k2 replaces k7
  
    err = 0.0;
    //if (ArrayIdx==11429) {
    //  printf("rk5sfact: %d %f %f %f %f %f %f %f\n", ArrayIdx,
    //     h, k1[0], k3[0], k4[0], k5[0], k6[0], k2[0]);    
    //  printf("rk5snum: %d %f\n", ArrayIdx,
    //   h*(e1*k1[0] + e3*k3[0] + e4*k4[0] + e5*k5[0] + e6*k6[0] + e7*k2[0]));
    //  printf("rk5sdenom: %d %f\n", ArrayIdx,
    //     abs_tol + rel_tol*MAX(fabs(y[0]), fabs(y_new[0])));	     
    //}
    for (int i=0; i<NVAR; i++) {
      float val = h*(e1*k1[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i]
		     + e7*k2[i])
	/ (abs_tol + rel_tol*MAX(fabs(y[i]), fabs(y_new[i])));
      //if (ArrayIdx==11429) {
      //	printf("rk5sval: %d %d %f\n", ArrayIdx, i, val);
      //}

      err += val*val;
    }
    err = sqrt(err/NVAR);

    float x_new = x + h;
    bool rejected=false;

    if (err<min_err) err = min_err;
    if (err>max_err) err = max_err;
    float fact=coeff*pow(err,-alpha);
    if (rejected && fact>1.0) fact=1.0;
    h *= fact;

    if (h <= h_min) {
      h = h_min;
      rejected = false;
    }
    else if (err <= 1.0) rejected = false; 
    else rejected = true;
    //TMP
    //int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
    //if (ArrayIdx==11429) {
    //  printf("rk5se: %d %f %f %f %f %f\n", ArrayIdx, x, x_new, h, fact, err);
    //}
      //
    if (!rejected) {
      x = x_new;
      break;
    }
  }
  
  for (int i=0; i<NVAR; i++) {
    y[i] = y_new[i];
  }
}

template<int NVAR, int NPARAMS, class DataStruct>
__device__
void RK5Update(float &x, float *y, float x1, float &h, float h_min,
	       float *params, DataStruct data_struct)
{
  bool end_time_step=false;
  while(!end_time_step) {
    float hmax=x1-x;
    RK5Step<NVAR, NPARAMS, DataStruct>(x, y, h, h_min, hmax, params,
				       data_struct);
    end_time_step = (x >= x1-h_min);
    ExternalUpdate<NVAR, NPARAMS, DataStruct>(x, y, params, end_time_step,
					      data_struct);
  }
}

template<int NVAR, int NPARAMS, class DataStruct>
__global__
void ArrayUpdate(int array_size, float *x_arr, float *h_arr, float *y_arr,
		 float *par_arr, float x1, float h_min, DataStruct data_struct)
{
  int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (ArrayIdx<array_size) {
    float x = x_arr[ArrayIdx];
    float h = h_arr[ArrayIdx];
    float y[NVAR];
    float params[NPARAMS];

    for(int i=0; i<NVAR; i++) {
      y[i] = y_arr[ArrayIdx*NVAR + i];
    }
    for(int j=0; j<NPARAMS; j++) {
      params[j] = par_arr[ArrayIdx*NPARAMS + j];
    }

    RK5Update<NVAR, NPARAMS, DataStruct>(x, y, x1, h, h_min, params,
					 data_struct);

    //float poisson_weight = 1.187*PoissonData[0];
    //if (poisson_weight>0) HandleSpike(poisson_weight, 0, y, params);

    x_arr[ArrayIdx] = x;
    h_arr[ArrayIdx] = h;
    for(int i=0; i<NVAR; i++) {
      y_arr[ArrayIdx*NVAR + i] = y[i];
    }
       
  }
}

template<class DataStruct>
class RungeKutta5
{
  int array_size_;
  int n_var_;
  int n_params_;
    
  float *d_XArr;
  float *d_HArr;
  float *d_YArr;
  float *d_ParamsArr;

  public:

  ~RungeKutta5();
 
  float *GetXArr() {return d_XArr;}
  float *GetHArr() {return d_HArr;}
  float *GetYArr() {return d_YArr;}
  float *GetParamsArr() {return d_ParamsArr;}
  int Init(int array_size, int n_var, int n_params, float x_min, float h);
  int Calibrate(float x_min, float h);

  int Free();

  int GetX(int i_array, int n_elems, float *x);
  int GetY(int i_var, int i_array, int n_elems, float *y);
  int SetParams(int i_param, int i_array, int n_params, int n_elems, float val);
  int SetVectParams(int i_param, int i_array, int n_params, int n_elems,
		    float *params, int vect_size);
  template<int NVAR, int NPARAMS> int Update(float x1, float h_min,
					     DataStruct data_struct);

};


template<class DataStruct>
template<int NVAR, int NPARAMS>
  int RungeKutta5<DataStruct>::Update(float x1, float h_min,
				      DataStruct data_struct)
{
  ArrayUpdate<NVAR, NPARAMS, DataStruct><<<(array_size_+1023)/1024, 1024>>>
    (array_size_, d_XArr, d_HArr, d_YArr, d_ParamsArr, x1, h_min, data_struct);
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
  cudaFree(d_ParamsArr);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Init(int array_size, int n_var, int n_params,
				  float x_min, float h)
{
  array_size_ = array_size;
  n_var_ = n_var;
  n_params_ = n_params; 

  gpuErrchk(cudaMalloc(&d_XArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_HArr, array_size_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_YArr, array_size_*n_var_*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_ParamsArr, array_size_*n_params_*sizeof(float)));

  //ArrayDef<<<1, 1>>>(array_size_, d_XArr, d_HArr, d_YArr, d_ParamsArr);
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );

  //ArrayAlloc();

  ArrayInit<<<(array_size+1023)/1024, 1024>>>(array_size_, n_var, n_params,
    d_XArr, d_HArr, d_YArr, d_ParamsArr, x_min, h);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::Calibrate(float x_min, float h)
{
  ArrayCalibrate<<<(array_size_+1023)/1024, 1024>>>(array_size_, n_var_,
     n_params_, d_XArr, d_HArr, d_YArr, d_ParamsArr, x_min, h);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetX(int i_array, int n_elems, float *x)
{
  cudaMemcpy(x, &d_XArr[i_array], n_elems*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::GetY(int i_var, int i_array, int n_elems, float *y)
{
  cudaMemcpy(y, &d_YArr[i_array*n_var_ + i_var], n_elems*sizeof(float),
	     cudaMemcpyDeviceToHost);

  return 0;
}

template<class DataStruct>
int RungeKutta5<DataStruct>::SetParams(int i_param, int i_array, int n_params,
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

template<class DataStruct>
int RungeKutta5<DataStruct>::SetVectParams(int i_param, int i_array,
					   int n_params, int n_elems,
					   float *params, int vect_size)
{
  for (int i=0; i<vect_size; i++) {
    SetFloatArray<<<(n_elems+1023)/1024, 1024>>>
      (&d_ParamsArr[i_array*n_params_ + N0_PARAMS + i_param + 4*i], n_elems,
       n_params, params[i]);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  return 0;
}

#endif
