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

#include <iostream>
#include <string>

#include "cuda_error.h"
#include "neuron_models.h"
#include "neuralgpu.h"
#include "aeif.h"

Nodes NeuralGPU::CreateNeuron(std::string model_name, int n_neurons, int n_receptors)
{
  CheckUncalibrated("Neurons cannot be created after calibration");
   if (n_neurons <= 0) {
    std::cerr << "Number of neurons must be greater than zero.\n";
    exit(0);
  }
  else if (n_receptors <= 0) {
    std::cerr << "Number of receptors must be greater than zero.\n";
    exit(0);
  }
  if (model_name == neuron_model_name[i_AEIF_model]) {
    AEIF *aeif_neuron = new AEIF;
    neuron_vect_.push_back(aeif_neuron);
  }
  else {
    std::cerr << "Unknown neuron model name: " << model_name << std::endl;
    exit(0);
  }
  return Nodes(CreateNeuron(n_neurons, n_receptors), n_neurons);
}

