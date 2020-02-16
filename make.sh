#!/bin/bash
if [ "$#" -ne 0 ] && ([ "$#" -ne 2 ] || [ "$1" != "--dir" ]); then
    echo
    echo "Usage: ./make.sh"
    echo "The default installation folder is $HOME/neurongpu"
    echo "If you want to install the library in a different folder, do:"
    echo "./make.sh [--dir installation-folder]"
    echo
    exit
fi
ver=$(head -1 VERSION | tr -d '\040\011\012\015')
if [ "$#" -eq 2 ]; then
    instdir=$2/neurongpu-$ver
else
    instdir=$HOME/neurongpu/neurongpu-$ver
fi
libdir=$instdir/lib
mkdir -p $libdir
if [ ! -d "$libdir" ]; then
    echo
    echo "Error:"
    echo "Cannot create the directory $libdir for installing the library."
    exit
fi
bindir=$instdir/bin
mkdir -p $bindir
if [ ! -d "$bindir" ]; then
    echo
    echo "Error:"
    echo "Cannot create the directory $bindir for installing the binaries."
    exit
fi
pydir=$instdir/python
mkdir -p $pydir
if [ ! -d "$pydir" ]; then
    echo
    echo "Error:"
    echo "Cannot create the directory $pydir for installing the python library."
    exit
fi
cppdir=$instdir/c++
mkdir -p $cppdir
if [ ! -d "$cppdir" ]; then
    echo
    echo "Error:"
    echo "Cannot create the directory $cppdir for installing the c++ examples."
    exit
fi

cd src

echo "Compiling the library"
nvcc -ccbin=mpic++ --compiler-options '-O3 -Wall -fPIC -fopenmp' -arch sm_30 --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true --shared -o ../lib/libneurongpu.so aeif_cond_alpha.cu aeif_psc_alpha.cu aeif_psc_exp.cu aeif_psc_delta.cu stdp.cu syn_model.cu test_syn_model.cu neurongpu.cu nested_loop.cu rev_spike.cu spike_buffer.cu connect.cu user_m1.cu user_m2.cu aeif_cond_beta.cu rk5.cu neuron_models.cu spike_detector.cu parrot_neuron.cu spike_generator.cu poiss_gen.cu base_neuron.cu connect_rules.cpp scan.cu connect_mpi.cu poisson.cu send_spike.cu get_spike.cu spike_mpi.cu getRealTime.cu multimeter.cu random.cu prefix_scan.cu node_group.cu -lm -lcurand

if [ $? -ne 0 ]; then
    cd ..
    echo "Error compiling the library"
    exit
fi
cd ..
echo
echo "Compiling the C wrapper (also needed for using the python interface"
g++ -Wall -fPIC -shared -L ./lib -I ./src -o lib/libneurongpu_C.so src/neurongpu_C.cpp -lneurongpu

if [ $? -ne 0 ]; then
    echo "Error compiling the C wrapper"
    exit
fi

cp lib/libneurongpu.so $libdir
cp lib/libneurongpu_C.so $libdir
cp -r python/* $pydir
cp -r c++/* $cppdir
cp -r src $instdir
cat bin/neurongpu_env.sh | sed "s:__instdir__:$instdir:" > $instdir/bin/neurongpu_env.sh

. $instdir/bin/neurongpu_env.sh
echo
echo "Done"

