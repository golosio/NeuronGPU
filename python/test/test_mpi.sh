mpi_pass_str[0]="MPI TEST PASSED"
mpi_pass_str[1]="MPI TEST NOT PASSED"
:>log.txt
for fn in test_brunel_mpi.py test_brunel_outdegree_mpi.py; do
    mpirun -np 2 python $fn 2>&1 | grep -v dyl >>log.txt
    res=$?
    echo ${mpi_pass_str[$res]}
done
