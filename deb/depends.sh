pkg_list="nvidia-cuda-toolkit openmpi-bin libopenmpi-dev libomp-dev python python-matplotlib python-mpi4py"
apt list $pkg_list 2>/dev/null | tr '/' ' ' | grep -v Listing | while read a b c d; do echo -n "$a (>= $c), "; done | sed 's/, $//'; echo
