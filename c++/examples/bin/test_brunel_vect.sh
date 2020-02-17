export OMP_NUM_THREADS=1
./brunel_net 1000
./brunel_vect 1000
diff test_brunel_vect.dat test_brunel_net.dat
unset OMP_NUM_THREADS
