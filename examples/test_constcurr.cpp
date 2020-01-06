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
  
  //////////////////////////////////////////////////////////////////////
  // WRITE HERE COMMANDS THAT ARE EXECUTED ON ALL HOSTS
  //////////////////////////////////////////////////////////////////////

  srand(12345);
  int n_neurons = 10000;
  
  // each host has n_neurons neurons with 3 receptor ports
  int neuron = neural_gpu.CreateNeuron(n_neurons, 1);

  // the following parameters are set to the same values on all hosts
  neural_gpu.SetNeuronParams("a", neuron, n_neurons,  4.0);
  neural_gpu.SetNeuronParams("b", neuron, n_neurons,  80.5);
  neural_gpu.SetNeuronParams("E_L", neuron, n_neurons,  -70.6);
  neural_gpu.SetNeuronParams("I_e", neuron, n_neurons,  700.0);

  string filename = "test_constcurr.dat";
  int i_neurons[] = {0}; // any set of neuron indexes
  // create multimeter record of V_m
  neural_gpu.CreateRecord(filename, "V_m", i_neurons, 1);

  neural_gpu.Simulate();

  neural_gpu.MpiFinalize();

  return 0;
}
