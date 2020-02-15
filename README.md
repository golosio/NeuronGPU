# NeuralGPU
A GPU-MPI library for simulation of large-scale networks of spiking neurons.
Can be used in Python, in C++ and in C.

*With this library it is possible to run relatively fast simulations of large-scale networks of spiking neurons. For instance, on a single Nvidia GeForce RTX 2080 Ti GPU board it is possible to simulate the activity of 1 million multisynapse AdEx neurons with 1000 synapse per neurons, for a total of 1 billion synapse, in little more than 100 seconds per second of neural activity.
The MPI communication is also very efficient.
The python interface is very similar to that of the NEST simulator: the most used commands are practically identical, dictionaries are used to define neurons, connections and synapsis properties in the same way.

For the installation, follow the instructions in the file INSTALLATION

To start using it,, have a look at the examples in the python/examples and c++/examples folders.


# NeuralGPU software specifications
* Simulated neuron models: different multisynapse AdEx models with current or conductance based synapses, and two user definable models.
* Simulated devices: Poisson signal generator, spike generator, multimeter, spike detector, parrot neuron.
* Synapse model: standard synapse, STDP (spike timing dependent plasticity), test synapse model
* Synaptic current models: conductance based models with synaptic rise time and decay time described by the alpha or by the beta function, current based models with synaptic rise time and decay time described by exp, alpha or delta functions. 
* Differential equation integration method: 5th order Runge-Kutta with adaptive step-size control.
* Connection rules: one-to-one, all-to-all, fixed indegree, fixed outdegree, fixed total number, user defined.
* Possibility to use arrays and distributions for connection weights and delays 
* GPU Cluster: efficient implementation of GPU-MPI.
* Numeric precision: 32 bit float.
* Simulation real time: on a machine with a single Nvidia GeForce RTX 2080 Ti GPU board, NeuralGPU is about 20 times faster than CPU code running on a PC with a CPU Intel CORE I9-9900K with with a frequency of 3.6GHz and 16 hardware threads.

