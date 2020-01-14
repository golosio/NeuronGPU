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
  int mpi_np = neural_gpu.MpiNp();
  if (mpi_np != 2) {
    cerr << "Usage: mpirun -np 2 brunel_mpi\n";
    return -1;
  }
  
  int mpi_id = neural_gpu.MpiId();
  cout << "Building on host " << mpi_id << " ..." <<endl;

  neural_gpu.SetMaxSpikeBufferSize(10); // max spike buffer size per neuron
  
  //////////////////////////////////////////////////////////////////////
  // WRITE HERE COMMANDS THAT ARE EXECUTED ON ALL HOSTS
  //////////////////////////////////////////////////////////////////////
  int n_receptors = 2;

  float delay = 1.0;       // synaptic delay in ms

  int order = 2500;
  int NE = 4 * order;      // number of excitatory neurons
  int NI = 1 * order;      // number of inhibitory neurons
  int n_neurons = NE + NI; // number of neurons in total

  int CE = 800;  // number of excitatory synapses per neuron
  int CI = CE/4;  // number of inhibitory synapses per neuron

  float Wex = 0.05;
  float Win = 0.35;

  // each host has a poisson generator
  float poiss_rate = 20000.0; // poisson signal rate in Hz
  float poiss_weight = 0.37;
  float poiss_delay = 0.2; // poisson signal delay in ms
  int n_pg = n_neurons; // number of poisson generators
  // create poisson generator
  NodeSeq pg = neural_gpu.CreatePoissonGenerator(n_pg, poiss_rate);

  // each host has n_neurons neurons with n_receptor receptor ports
  NodeSeq neuron = neural_gpu.CreateNeuron("aeif_cond_beta", n_neurons,
					   n_receptors);
  NodeSeq exc_neuron = neuron.Subseq(0,NE-1); // excitatory neuron group
  NodeSeq inh_neuron = neuron.Subseq(NE, n_neurons-1); //inhibitory neuron group
  
  // the following parameters are set to the same values on all hosts
  float E_rev[] = {0.0, -85.0};
  float taus_decay[] = {1.0, 1.0};
  float taus_rise[] = {1.0, 1.0};
  neural_gpu.SetNeuronParam("E_rev", neuron, E_rev, 2);
  neural_gpu.SetNeuronParam("taus_decay", neuron, taus_decay, 2);
  neural_gpu.SetNeuronParam("taus_rise", neuron, taus_rise, 2);

  // Excitatory local connections, defined on all hosts
  // connect excitatory neurons to port 0 of all neurons
  // weight Wex and fixed indegree CE*3/4
  ConnSpec conn_spec1(FIXED_INDEGREE, CE*3/4);
  SynSpec syn_spec1;
  syn_spec1.SetParam("receptor", 0);
  syn_spec1.SetParam("weight", Wex);
  syn_spec1.SetParam("delay", delay);
  neural_gpu.Connect(exc_neuron, neuron, conn_spec1, syn_spec1);

  // Inhibitory local connections, defined on all hosts
  // connect inhibitory neurons to port 1 of all neurons
  // weight Win and fixed indegree CI*3/4
  ConnSpec conn_spec2(FIXED_INDEGREE, CI*3/4);
  SynSpec syn_spec2;
  syn_spec2.SetParam("receptor", 1);
  syn_spec2.SetParam("weight", Win);
  syn_spec2.SetParam("delay", delay);
  neural_gpu.Connect(inh_neuron, neuron, conn_spec2, syn_spec2);

  ConnSpec conn_spec3(ONE_TO_ONE);
  SynSpec syn_spec3(STANDARD_SYNAPSE, poiss_weight, poiss_delay, 0);
  // connect poisson generator to port 0 of all neurons
  neural_gpu.Connect(pg, neuron, conn_spec3, syn_spec3);


  char filename[100];
  sprintf(filename, "test_brunel_mpi_%d.dat", mpi_id);

  int i_neuron_arr[] = {neuron[0], neuron[rand()%n_neurons],
		     neuron[n_neurons-1]}; // any set of neuron indexes
  // create multimeter record of V_m
  std::string var_name_arr[] = {"V_m", "V_m", "V_m"};
  neural_gpu.CreateRecord(string(filename), var_name_arr, i_neuron_arr, 3);

  //////////////////////////////////////////////////////////////////////
  // WRITE HERE REMOTE CONNECTIONS
  //////////////////////////////////////////////////////////////////////

  // Excitatory remote connections
  // connect excitatory neurons to port 0 of all neurons
  // weight Wex and fixed indegree CE-CE*3/4
  // host 0 to host 1
  ConnSpec conn_spec4(FIXED_INDEGREE, CE-CE*3/4);
  SynSpec syn_spec4;
  syn_spec4.SetParam("receptor", 0);
  syn_spec4.SetParam("weight", Wex);
  syn_spec4.SetParam("delay", delay);
  neural_gpu.RemoteConnect(0, exc_neuron, 1, neuron, conn_spec4, syn_spec4);

  // host 1 to host 0
  neural_gpu.RemoteConnect(1, exc_neuron, 0, neuron, conn_spec4, syn_spec4);

  // Inhibitory remote connections
  // connect inhibitory neurons to port 1 of all neurons
  // weight Win and fixed indegree CI-CI*3/4
  // host 0 to host 1

  
  ConnSpec conn_spec5(FIXED_INDEGREE, CI-CI*3/4);
  SynSpec syn_spec5;
  syn_spec5.SetParam("receptor", 1);
  syn_spec5.SetParam("weight", Win);
  syn_spec5.SetParam("delay", delay);
  neural_gpu.RemoteConnect(0, inh_neuron, 1, neuron, conn_spec5, syn_spec5);

  // host 1 to host 0
  neural_gpu.RemoteConnect(1, inh_neuron, 0, neuron, conn_spec5, syn_spec5);

  // just to have same results in different simulations
  neural_gpu.SetRandomSeed(1234ULL);
  neural_gpu.Simulate();

  neural_gpu.MpiFinalize();

  return 0;
}
