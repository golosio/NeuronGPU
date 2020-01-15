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

#include "ngpu_exception.h"
#include "cuda_error.h"
#include "neuron_models.h"
#include "neuralgpu.h"
#include "aeif_cond_beta.h"

NodeSeq NeuralGPU::CreateNeuron(std::string model_name, int n_nodes, int n_ports)
{
  CheckUncalibrated("Nodes cannot be created after calibration");
   if (n_nodes <= 0) {
     throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  else if (n_ports <= 0) {
    throw ngpu_exception("Number of ports must be greater than zero.");
  }
  if (model_name == neuron_model_name[i_aeif_cond_beta_model]) {
    aeif_cond_beta *aeif_cond_beta_neuron = new aeif_cond_beta;
    node_vect_.push_back(aeif_cond_beta_neuron);
  }
  else {
    throw ngpu_exception(std::string("Unknown neuron model name: ")
			 + model_name);
  }
  return NodeSeq(CreateNodeGroup(n_nodes, n_ports), n_nodes);
}

