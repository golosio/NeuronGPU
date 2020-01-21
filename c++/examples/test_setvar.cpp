/*
Copyright (C) 2020 Bruno Golosio
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
  NeuralGPU ngpu;
  cout << "Building ...\n";
  
  srand(12345);
  int n_neurons = 3;
  
  // create n_neurons neurons with 2 receptor ports
  NodeSeq neuron = ngpu.Create("aeif_cond_beta", n_neurons, 2);
  float taus_decay[] = {60.0, 10.0};
  float taus_rise[] = {40.0, 5.0};
  ngpu.SetNeuronParam(neuron, "taus_decay", taus_decay, 2);
  ngpu.SetNeuronParam(neuron, "taus_rise", taus_rise, 2);
  
  NodeSeq neuron0 = neuron.Subseq(0,0);
  NodeSeq neuron1 = neuron.Subseq(1,1);
  NodeSeq neuron2 = neuron.Subseq(2,2);
  float g11[] = {0.0, 0.1};
  float g12[] = {0.1, 0.0};
  
  // neuron variables
  ngpu.SetNeuronVar(neuron0, "V_m", -80.0);
  ngpu.SetNeuronVar(neuron1, "g1", g11, 2);
  ngpu.SetNeuronVar(neuron2, "g1", g12, 2);
  
  string filename = "test_setvar.dat";
  int i_neurons[] = {neuron[0], neuron[1], neuron[2]};
  string var_name[] = {"V_m", "V_m", "V_m"};

  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neurons, 3);

  ngpu.Simulate();

  return 0;
}
