NVCC=nvcc
COMPILER_FLAGS= -O3 -Wall -fPIC -fopenmp
CUDA_FLAGS= -arch sm_30 --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true
CUDA_LIBS= -lm -lcurand
CUDA_LDFLAGS = -shared   # linking flags

.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

CXXLD=$(NVCC)

lib_LTLIBRARIES = libneurongpu.la
libneurongpu_la_SOURCES = dummyfile.cpp
libneurongpu_la_LDFLAGS = $(CUDA_LDFLAGS)
libneurongpu_la_CFLAGS = $(CUDA_FLAGS) $(COMPILER_FLAGS)
libneurongpu_la_LIBADD = $(CUDA_LIBS)
