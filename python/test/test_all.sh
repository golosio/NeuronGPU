pass_str[0]=PASSED
pass_str[1]="NOT PASSED"
:>log.txt
for fn in test_stdp_list.py test_stdp.py test_syn_model.py test_brunel_list.py test_brunel_outdegree.py test_brunel_user_m1.py test_spike_detector.py; do
    python $fn >>log.txt
    res=$?
    echo ${pass_str[$res]}
done
for fn in test_brunel_mpi.py test_brunel_outdegree_mpi.py; do
    mpirun -np 2 python $fn >>log.txt
    res=$?
    echo ${pass_str[$res]}
done
for fn in syn_group connect getarr setvar2 setvar3; do
    python test_$fn.py > tmp
    diff -qs tmp log_$fn.txt >> log.txt
    res=$?
    echo ${pass_str[$res]}    
done
rm tmp
