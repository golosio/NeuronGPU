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
#include "poiss_gen.h"

NodeSeq NeuralGPU::Create(std::string model_name, int n_nodes /*=1*/,
			  int n_ports /*=1*/)
{
  CheckUncalibrated("Nodes cannot be created after calibration");
   if (n_nodes <= 0) {
     throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  else if (n_ports < 0) {
    throw ngpu_exception("Number of ports must be >= zero.");
  }
  if (model_name == neuron_model_name[i_aeif_cond_beta_model]) {
    aeif_cond_beta *aeif_cond_beta_group = new aeif_cond_beta;
    node_vect_.push_back(aeif_cond_beta_group);
  }
  else if (model_name == neuron_model_name[i_poisson_generator_model]) {
    n_ports = 0;
    poiss_gen *poiss_gen_group = new poiss_gen;
    node_vect_.push_back(poiss_gen_group);
  }
  else {
    throw ngpu_exception(std::string("Unknown neuron model name: ")
			 + model_name);
  }
  return NodeSeq(CreateNodeGroup(n_nodes, n_ports), n_nodes);
}

