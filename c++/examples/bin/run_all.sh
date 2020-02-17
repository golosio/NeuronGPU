mpirun -np 2 brunel_mpi 2000
./brunel_net 2000
./brunel_vect 2000
./brunel_outdegree 2000
mpirun -np 2 ./brunel_outdegree_mpi 2000
./test_aeif_cond_beta
./test_constcurr
