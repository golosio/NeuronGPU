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
#include <stdlib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "neural_gpu.h"

using namespace std;

int main(int argc, char *argv[])
{
  // Intializes C random number generator
  // srand((unsigned) time(&t));

  NeuralGPU neural_gpu;
  neural_gpu.ConnectMpiInit(argc, argv);
  int mpi_id = neural_gpu.MpiId();
  cout << "Building on host " << mpi_id << " ..." <<endl;

  neural_gpu.max_spike_buffer_num_=10; //reduce it to save GPU memory
  
  //////////////////////////////////////////////////////////////////////
  // WRITE HERE COMMANDS THAT ARE EXECUTED ON ALL HOSTS
  //////////////////////////////////////////////////////////////////////


  // poisson generator parameters
  float poiss_rate = 5000.0; // poisson signal rate in Hz
  float poiss_weight = 1.0;
  float poiss_delay = 0.2; // poisson signal delay in ms
  int n_pg = 7; // number of poisson generators
  // create poisson generator
  int pg = neural_gpu.CreatePoissonGenerator(n_pg, poiss_rate);

  int n_recept = 3; // number of receptors
  // create 3 neuron groups
  int n_neur1 = 100; // number of neurons
  int n_neur2 = 20;
  int n_neur3 = 50;
  int n_neurons = n_neur1 + n_neur2 + n_neur3;
  
  int neur_group1 = neural_gpu.CreateNeuron("AEIF", n_neurons, n_recept);
  int neur_group2 = neur_group1 + n_neur1;
  int neur_group3 = neur_group2 + n_neur2;
  
  // the following parameters are set to the same values on all hosts
  float E_rev[] = {0.0, 0.0, 0.0};
  float taus_decay[] = {1.0, 1.0, 1.0};
  float taus_rise[] = {1.0, 1.0, 1.0};
  neural_gpu.SetNeuronVectParams("E_rev", neur_group1, n_neur1, E_rev, 3);
  neural_gpu.SetNeuronVectParams("taus_decay", neur_group1, n_neur1,
				 taus_decay, 3);
  neural_gpu.SetNeuronVectParams("taus_rise", neur_group1, n_neur1,
				 taus_rise, 3);
  neural_gpu.SetNeuronVectParams("E_rev", neur_group2, n_neur2, E_rev, 3);
  neural_gpu.SetNeuronVectParams("taus_decay", neur_group2, n_neur2,
				 taus_decay, 3);
  neural_gpu.SetNeuronVectParams("taus_rise", neur_group2, n_neur2,
				 taus_rise, 3);
  neural_gpu.SetNeuronVectParams("E_rev", neur_group3, n_neur3, E_rev, 3);
  neural_gpu.SetNeuronVectParams("taus_decay", neur_group3, n_neur3,
				 taus_decay, 3);
  neural_gpu.SetNeuronVectParams("taus_rise", neur_group3, n_neur3,
				 taus_rise, 3);

  int i11 = neur_group1 + rand()%n_neur1;
  int i12 = neur_group2 + rand()%n_neur2;
  int i13 = neur_group2 + rand()%n_neur2;
  int i14 = neur_group3 + rand()%n_neur3;

  int i21 = neur_group2 + rand()%n_neur2;

  int i31 = neur_group1 + rand()%n_neur1;
  int i32 = neur_group3 + rand()%n_neur3;

  int it1 = neur_group1 + rand()%n_neur1;
  int it2 = neur_group2 + rand()%n_neur2;
  int it3 = neur_group3 + rand()%n_neur3;
  
  // connect poisson generator to port 0 of all neurons
  neural_gpu.Connect(pg, i11, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+1, i12, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+2, i13, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+3, i14, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+4, i21, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+5, i31, 0, poiss_weight, poiss_delay);
  neural_gpu.Connect(pg+6, i32, 0, poiss_weight, poiss_delay);

  float weight = 0.01; // connection weight
  float delay = 0.2; // connection delay in ms

  // connect neurons to target neuron n. 1
  neural_gpu.Connect(i11, it1, 0, weight, delay);
  neural_gpu.Connect(i12, it1, 1, weight, delay);
  neural_gpu.Connect(i13, it1, 1, weight, delay);
  neural_gpu.Connect(i14, it1, 2, weight, delay);

  // connect neuron to target neuron n. 2
  neural_gpu.Connect(i21, it2, 0, weight, delay);

    // connect neurons to target neuron n. 3
  neural_gpu.Connect(i31, it3, 0, weight, delay);
  neural_gpu.Connect(i32, it3, 1, weight, delay);
  
  // create multimeter record n.1
  char filename1[] = "test_connections_voltage.dat";
  int i_neuron_arr1[] = {i11, i12, i13, i14, i21, i31, i32, it1, it2, it3};
  std::string var_name_arr1[] = {"V_m", "V_m", "V_m", "V_m", "V_m", "V_m",
				"V_m", "V_m", "V_m", "V_m"};
  neural_gpu.CreateRecord(string(filename1), var_name_arr1, i_neuron_arr1, 10);

  // create multimeter record n.2
  char filename2[] = "test_connections_g1.dat";
  int i_neuron_arr2[] = {it1, it1, it1, it2, it3, it3};
  int i_receptor_arr[] = {0, 1, 2, 0, 0, 1};
  std::string var_name_arr2[] = {"g1", "g1", "g1", "g1", "g1", "g1"};
  neural_gpu.CreateRecord(string(filename2), var_name_arr2, i_neuron_arr2,
			  i_receptor_arr, 6);

  neural_gpu.SetRandomSeed(1234ULL);
  neural_gpu.Simulate();

  neural_gpu.MpiFinalize();

  return 0;
}
