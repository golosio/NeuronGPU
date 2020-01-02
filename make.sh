if [ "$#" -ne 1 ]; then
    echo
    echo "Please provide the path for installing the library."
    echo "Example:"
    echo "./make.sh $HOME/lib/"
    exit
fi

if [ ! -d "$1" ]; then
    mkdir $1

    if [ ! -d "$1" ]; then
	echo
	echo "Error:"
	echo "Cannot create the directory $1 for installing the library."
	exit
    fi
fi

cd src

nvcc -ccbin=mpicc --compiler-options -Wall --compiler-options '-fPIC' --compiler-options '-fopenmp' -arch sm_30 --ptxas-options=-v --maxrregcount=55 --relocatable-device-code true --shared -o ../lib/libneuralgpu.so scan.cu neuron_models.cu base_neuron.cu neural_gpu.cu aeif.cu connect.cu connect_mpi.cu poisson.cu rk5.cu spike_buffer.cu send_spike.cu get_spike.cu spike_mpi.cu getRealTime.cu spike_generator.cu multimeter.cu random.cu nested_loop.cu prefix_scan.cu neuron_group.cu -lm -lstdc++ -lcurand

cd ..

cp lib/libneuralgpu.so $1

echo
echo "Done"
echo "Please be sure to include the directory $1 in your shared library path"
echo "For instance in Linux you can add to your .bashrc file"
echo "the following line:"
echo "export LD_LIBRARY_PATH=$1:\$LD_LIBRARY_PATH"
