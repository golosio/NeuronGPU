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
  
  ngpu.SetSimTime(300.0); // simulation time in ms
  
  srand(12345);
  int n_neurons = 10000;
  
  // each host has n_neurons neurons with 3 receptor ports
  NodeSeq neuron = ngpu.CreateNeuron("AEIF", n_neurons, 3);

  // the following parameters are set to the same values on all hosts
  float E_rev[] = {20.0, 0.0, -85.0};
  float taus_decay[] = {40.0, 20.0, 30.0};
  float taus_rise[] = {20.0, 10.0, 5.0};
  ngpu.SetNeuronParam("E_rev", neuron, E_rev, 3);
  ngpu.SetNeuronParam("taus_decay", neuron, taus_decay, 3);
  ngpu.SetNeuronParam("taus_rise", neuron, taus_rise, 3);
  ngpu.SetNeuronParam("a", neuron,  4.0);
  ngpu.SetNeuronParam("b", neuron,  80.5);
  ngpu.SetNeuronParam("E_L", neuron,  -70.6);
  ngpu.SetNeuronParam("g_L", neuron,  300.0);

  int n_sg = 1; // number of spike generators
  NodeSeq sg = ngpu.CreateSpikeGenerator(n_sg); // create spike generator

  float spike_time[] = {10.0};
  float spike_height[] = {1.0};
  int n_spikes = 1;
  int sg_node = 0; // this spike generator has only one node
  // set spike times and height
  ngpu.SetSpikeGenerator(sg_node, n_spikes, spike_time, spike_height);
  float delay[] = {1.0, 100.0, 130.0};
  float weight[] = {0.1, 0.2, 0.15};

  for (int i_port=0; i_port<3; i_port++) {
    ConnSpec conn_spec(ALL_TO_ALL);
    SynSpec syn_spec(STANDARD_SYNAPSE, weight[i_port], delay[i_port], i_port);
    ngpu.Connect(sg, neuron, conn_spec, syn_spec);
  }

  string filename = "test_aeif.dat";
  int i_neuron[] = {neuron[rand()%n_neurons]}; // any set of neuron indexes
  string var_name[] = {"V_m"};
  // create multimeter record of V_m
  ngpu.CreateRecord(filename, var_name, i_neuron, 1);

  ngpu.Simulate();

  return 0;
}
