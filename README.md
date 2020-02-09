# NeuralGPU
A GPU-MPI library for simulation of large scale networks of biological neurons.
Can be used in Python, in C++ and in C.

# NeuralGPU software specifications
* Simulated neuron models: multisynapse adaptive exponential integrate and fire model (AEIF, also called AdEx), and two user definable models.
* Simulated devices: Poisson signal generator, spike generator, multimeter, spike detector, parrot neuron.
* Synapse model: standard synapse, STDP (spike timing dependent plasticity), test synapse model
* Synaptic current model: conductance based model with independent rise time and decay time described by alpha or beta function.
* Differential equation integration method: 5th order Runge-Kutta with adaptive step-size control.
* Connection rules: one-to-one, all-to-all, fixed indegree, fixed outdegree, fixed total  number, user defined.
* Possibility to use arrays and distributions for connection weights and delays 
* GPU Cluster: efficient implementation of GPU-MPI.
* Numeric precision: 32 bit float.
* Simulation real time: on a machine with a single Nvidia GeForce RTX 2080 Ti GPU board, NeuralGPU is about 20 times faster than CPU code running on a PC with a CPU Intel CORE I9-9900K with with a frequency of 3.6GHz and 16 hardware threads.
* Installation requirements:
CUDA Toolkit
openmpi
openmpi-devel (libopenmpi-dev in ubuntu),

To install the library, from a terminal extract the source code and  type:
cd NeuralGPU
./make.sh /home/username/lib
where /home/username/lib should be replaced by the folder where you whish to install the library

To install the python interface type:
./make_C.sh /home/username/lib

Be sure to include the directory where you installed the library in your
shared library path. For instance in Linux you can add to your .bashrc file
the following line:
export LD_LIBRARY_PATH=/home/username/lib:\$LD_LIBRARY_PATH

To setup the environment for using the python interface, run the script
setenv.sh
which you can find in the python folder

To start, have a look at the examples in the python/examples and c++/examples
folders.
