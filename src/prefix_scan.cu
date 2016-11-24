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

#include <assert.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "scan_common.h"
#include "prefix_scan.h"

const uint PrefixScan::AllocSize = 13 * 1048576 / 2;

int PrefixScan::Init()
{
  /*
    printf("Allocating and initializing host arrays...\n");
    h_Input     = (uint *)malloc(AllocSize * sizeof(uint));
    h_OutputCPU = (uint *)malloc(AllocSize * sizeof(uint));
    h_OutputGPU = (uint *)malloc(AllocSize * sizeof(uint));

    printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input, AllocSize * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,
       AllocSize * sizeof(uint)));
  */
    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    return 0;
}

int PrefixScan::Scan(uint *d_Output, uint *d_Input, uint n)
{
   uint array_length = 1;
   while (array_length < n || array_length < MIN_SHORT_ARRAY_SIZE) {
      array_length <<= 1;
   }
   if (array_length > MAX_LARGE_ARRAY_SIZE) {
      fprintf(stderr, "Array length larger than maximum size "
                      "for prexix scan.\n");
      exit(-1);
   }
   if (array_length <= MAX_SHORT_ARRAY_SIZE) {
      scanExclusiveShort(d_Output, d_Input, AllocSize / array_length,
         array_length);
   }
   else {
      scanExclusiveLarge(d_Output, d_Input, AllocSize / array_length,
         array_length);
   }
   
   checkCudaErrors(cudaDeviceSynchronize());

   return 0;
}

int PrefixScan::Free()
{
   closeScan();
   //checkCudaErrors(cudaFree(d_Output));
   //checkCudaErrors(cudaFree(d_Input));

   return 0;
}
