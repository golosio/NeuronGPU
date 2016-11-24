nvcc -ccbin=mpicc --compiler-options -Wall -arch sm_30 --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true -L ../lib -I ../src -o ../bin/test_constcurr test_constcurr.cu -lm -lstdc++ -lneuralgpu

