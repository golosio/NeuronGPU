/*
Copyright (C) 2016 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "neuralgpu.h"

using namespace std;

int main(int argc, char *argv[])
{
  NeuralGPU neural_gpu;
  neural_gpu.ConnectMpiInit(argc, argv);
  cout << "Building on host " << neural_gpu.MpiId() << " ..." <<endl;
  
  neural_gpu.sim_time_=300; // simulation time in ms
  
  //////////////////////////////////////////////////////////////////////
  // WRITE HERE COMMANDS THAT ARE EXECUTED ON ALL HOSTS
  //////////////////////////////////////////////////////////////////////

  srand(12345);
  int n_neurons = 10000;
  
  // each host has n_neurons neurons with 3 receptor ports
  int neuron = neural_gpu.CreateNeuron(n_neurons, 3);

  // the following parameters are set to the same values on all hosts
  float E_rev[] = {20.0, 0.0, -85.0};
  float taus_decay[] = {40.0, 20.0, 30.0};
  float taus_rise[] = {20.0, 10.0, 5.0};
  neural_gpu.SetNeuronVectParams("E_rev", neuron, n_neurons, E_rev, 3);
  neural_gpu.SetNeuronVectParams("taus_decay", neuron, n_neurons,
				 taus_decay, 3);
  neural_gpu.SetNeuronVectParams("taus_rise", neuron, n_neurons, taus_rise, 3);
  neural_gpu.SetNeuronParams("a", neuron, n_neurons,  4.0);
  neural_gpu.SetNeuronParams("b", neuron, n_neurons,  80.5);
  neural_gpu.SetNeuronParams("E_L", neuron, n_neurons,  -70.6);
  neural_gpu.SetNeuronParams("g_L", neuron, n_neurons,  300.0);

  int n_sg = 1; // number of spike generators
  int sg = neural_gpu.CreateSpikeGenerator(n_sg); // create spike generator

  float spike_time[] = {10.0};
  float spike_height[] = {1.0};
  int n_spikes = 1;
  int sg_node = 0; // this spike generator has only one node
  // set spike times and height
  neural_gpu.SetSpikeGenerator(sg_node, n_spikes, spike_time, spike_height);
  float delay[] = {1.0, 100.0, 130.0};
  float weight[] = {0.1, 0.2, 0.15};

  for (int i_port=0; i_port<3; i_port++) {
    neural_gpu.ConnectAllToAll(sg, n_sg, neuron, n_neurons, i_port,
			       weight[i_port], delay[i_port]);
  }

  string filename = "test_aeif_multisynapse.dat";
  int i_neurons[] = {9990}; // any set of neuron indexes
  // create multimeter record of V_m
  neural_gpu.CreateRecord(filename, "V_m", i_neurons, 1);

  neural_gpu.Simulate();

  neural_gpu.MpiFinalize();

  return 0;
}
