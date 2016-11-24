# NeuralGPU
A GPU library for simulation of large scale networks of biological neurons

IMPORTANT NOTES: This is just a preliminary version of the library. You can try to use it under the term of the license, however we are not ready to accept pull requests. NeuralGPU is intended to be a library. For this reason it does not have a user interface or an interpreter. In the examples, the command are written in C++ files that must be compiled. I plan to propose an integration of NeuralGPU as an optional library in the NEST simulator (http://www.nest-simulator.org/).

# NeuralGPU software specifications
* Simulated neuron model: at the moment only the adaptive exponential integrate and fire model (AEIF), see below.
* Synaptic current model: conductance based model with independent rise time and decay time described by alpha or beta function.
* Differential equation integration method: 5th order Runge-Kutta with adaptive step-size control.
* Synapse model: standard synapse.
* Connection parameters: weight, delay, receptor port.
* Connection rules: one-to-one, all-to-all, fixed indegree, user defined.
* GPU Cluster: efficient implementation of GPU-MPI.
* Numeric precision: 32 bit float.
* Simulated devices: Poisson signal generator, spike generator, multimeter.
* Simulation real time: on a machine with a single GPU, NeuralGPU is about 20 times faster than CPU code running on an 8 core processor.

