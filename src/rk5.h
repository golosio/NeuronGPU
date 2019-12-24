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
extern __device__ int ARRAY_SIZE;
#include "rk5_const.h"
#include "derivatives.h"
#include "poisson.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

extern __device__ float *XArr;
extern __device__ float *HArr;
extern __device__ float *YArr;
extern __device__ float *ParamsArr;

template<int NVAR, int NPARAMS>
__device__
void RK5Update(float &x, float *y, float x1, float &h, float h_min,
	       float *params);

template<int NVAR, int NPARAMS>
__device__
void RK5Step(float &x, float *y, float &h, float h_min, float h_max,
	     float *params);

template<int NVAR, int NPARAMS>
__device__
void ExternalUpdate(float x, float *y, float *params, bool end_time_step);

__global__
void ArrayDef(int array_size, float *x_arr, float *h_arr, float *y_arr,
	      float *par_arr);

__global__
void ArrayInit(int n_var, int n_params, float x_min, float h);

__global__
void ArrayCalibrate(int n_var, int n_params, float x_min, float h);

__device__
void VarInit(int n_var, int n_params, float x, float *y, float *params);

__device__
void VarCalibrate(int n_var, int n_params, float x, float *y, float *params);

template<int NVAR, int NPARAMS>
__global__
void ArrayUpdate(float x1, float h_min);

template<int NVAR, int NPARAMS>
__device__
void Derivatives(float x, float *y, float *dydx, float *params);

template<int NVAR, int NPARAMS>
__device__
void RK5Step(float &x, float *y, float &h, float h_min, float h_max,
	     float *params)
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

    Derivatives<NVAR, NPARAMS>(x, y, k1, params);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*a21*k1[i];
    }
    Derivatives<NVAR, NPARAMS>(x+c2*h, y_new, k2, params);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a31*k1[i] + a32*k2[i]);
    }
    Derivatives<NVAR, NPARAMS>(x+c3*h, y_new, k3, params);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3[i]);
    }
    Derivatives<NVAR, NPARAMS>(x+c4*h, y_new, k4, params);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a51*k1[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
    }
    Derivatives<NVAR, NPARAMS>(x+c5*h, y_new, k5, params);
  
    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a61*k1[i] + a62*k2[i] + a63*k3[i] + a64*k4[i]
			  + a65*k5[i]);
    }
    float x1 = x + h;
    Derivatives<NVAR, NPARAMS>(x1, y_new, k6, params);

    for (int i=0; i<NVAR; i++) {
      y_new[i] = y[i] + h*(a71*k1[i] + a73*k3[i] + a74*k4[i] + a75*k5[i]
			  + a76*k6[i]);
    }
    Derivatives<NVAR, NPARAMS>(x1, y_new, k2, params); // k2 replaces k7
  
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

template<int NVAR, int NPARAMS>
__device__
void RK5Update(float &x, float *y, float x1, float &h, float h_min,
	       float *params)
{
  bool end_time_step=false;
  while(!end_time_step) {
    float hmax=x1-x;
    RK5Step<NVAR, NPARAMS>(x, y, h, h_min, hmax, params);
    end_time_step = (x >= x1-h_min);
    ExternalUpdate<NVAR, NPARAMS>(x, y, params, end_time_step);
  }
}

/*
template<int NVAR, int NPARAMS>
__global__
void ArrayUpdate(float x1, float h_min)
{
  int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (ArrayIdx<ARRAY_SIZE) {
    //float x = XArr[ArrayIdx];
    //float h = HArr[ArrayIdx];

    RK5Update<NVAR, NPARAMS>(XArr[ArrayIdx], &YArr[ArrayIdx*NVAR], x1,
			     HArr[ArrayIdx], h_min,
			     &ParamsArr[ArrayIdx*NPARAMS]);

    //float poisson_weight = 1.187*PoissonData[0];
    //if (poisson_weight>0) HandleSpike(poisson_weight, 0, y, params);

    //XArr[ArrayIdx] = x;
    //HArr[ArrayIdx] = h;
  }
}
*/

template<int NVAR, int NPARAMS>
__global__
void ArrayUpdate(float x1, float h_min)
{
  int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (ArrayIdx<ARRAY_SIZE) {
    float x = XArr[ArrayIdx];
    float h = HArr[ArrayIdx];
    float y[NVAR];
    float params[NPARAMS];

    for(int i=0; i<NVAR; i++) {
      y[i] = YArr[ArrayIdx*NVAR + i];
    }
    for(int j=0; j<NPARAMS; j++) {
      params[j] = ParamsArr[ArrayIdx*NPARAMS + j];
    }

    RK5Update<NVAR, NPARAMS>(x, y, x1, h, h_min, params);

    //float poisson_weight = 1.187*PoissonData[0];
    //if (poisson_weight>0) HandleSpike(poisson_weight, 0, y, params);

    XArr[ArrayIdx] = x;
    HArr[ArrayIdx] = h;
    for(int i=0; i<NVAR; i++) {
      YArr[ArrayIdx*NVAR + i] = y[i];
    }
       
  }
}

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
 
  int Init(int array_size, int n_var, int n_params, float x_min, float h);
  int Calibrate(float x_min, float h);

  int Free();

  int GetX(int i_array, int n_elems, float *x);
  int GetY(int i_var, int i_array, int n_elems, float *y);
  int SetParams(int i_param, int i_array, int n_params, int n_elems, float val);
  int SetVectParams(int i_param, int i_array, int n_params, int n_elems,
		    float *params, int vect_size);
};

#endif
