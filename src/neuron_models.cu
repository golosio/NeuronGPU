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

#include <config.h>
#include <iostream>
#include <string>

#include "ngpu_exception.h"
#include "cuda_error.h"
#include "neuron_models.h"
#include "neurongpu.h"
#include "iaf_psc_exp.h"
#include "ext_neuron.h"
#include "aeif_cond_beta.h"
#include "aeif_cond_alpha.h"
#include "aeif_psc_alpha.h"
#include "aeif_psc_delta.h"
#include "aeif_psc_exp.h"
#include "poiss_gen.h"
#include "spike_generator.h"
#include "parrot_neuron.h"
#include "spike_detector.h"
#include "user_m1.h"
#include "user_m2.h"

NodeSeq NeuronGPU::Create(std::string model_name, int n_node /*=1*/,
			  int n_port /*=1*/)
{
  CheckUncalibrated("Nodes cannot be created after calibration");
  if (n_node <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  else if (n_port < 0) {
    throw ngpu_exception("Number of ports must be >= zero.");
  }
  if (model_name == neuron_model_name[i_iaf_psc_exp_model]) {
    n_port = 2;
    iaf_psc_exp *iaf_psc_exp_group = new iaf_psc_exp;
    node_vect_.push_back(iaf_psc_exp_group);
  }
  else if (model_name == neuron_model_name[i_ext_neuron_model]) {
    ext_neuron *ext_neuron_group = new ext_neuron;
    node_vect_.push_back(ext_neuron_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_beta_model]) {
    aeif_cond_beta *aeif_cond_beta_group = new aeif_cond_beta;
    node_vect_.push_back(aeif_cond_beta_group);
  }
  else if (model_name == neuron_model_name[i_aeif_cond_alpha_model]) {
    aeif_cond_alpha *aeif_cond_alpha_group = new aeif_cond_alpha;
    node_vect_.push_back(aeif_cond_alpha_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_exp_model]) {
    aeif_psc_exp *aeif_psc_exp_group = new aeif_psc_exp;
    node_vect_.push_back(aeif_psc_exp_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_alpha_model]) {
    aeif_psc_alpha *aeif_psc_alpha_group = new aeif_psc_alpha;
    node_vect_.push_back(aeif_psc_alpha_group);
  }
  else if (model_name == neuron_model_name[i_aeif_psc_delta_model]) {
    n_port = 1;
    aeif_psc_delta *aeif_psc_delta_group = new aeif_psc_delta;
    node_vect_.push_back(aeif_psc_delta_group);
  }
  else if (model_name == neuron_model_name[i_user_m1_model]) {
    user_m1 *user_m1_group = new user_m1;
    node_vect_.push_back(user_m1_group);
  }
  else if (model_name == neuron_model_name[i_user_m2_model]) {
    user_m2 *user_m2_group = new user_m2;
    node_vect_.push_back(user_m2_group);
  }
  else if (model_name == neuron_model_name[i_poisson_generator_model]) {
    n_port = 0;
    poiss_gen *poiss_gen_group = new poiss_gen;
    node_vect_.push_back(poiss_gen_group);
  }
  else if (model_name == neuron_model_name[i_spike_generator_model]) {
    n_port = 0;
    spike_generator *spike_generator_group = new spike_generator;
    node_vect_.push_back(spike_generator_group);
  }
  else if (model_name == neuron_model_name[i_parrot_neuron_model]) {
    n_port = 2;
    parrot_neuron *parrot_neuron_group = new parrot_neuron;
    node_vect_.push_back(parrot_neuron_group);
  }
  else if (model_name == neuron_model_name[i_spike_detector_model]) {
    n_port = 1;
    spike_detector *spike_detector_group = new spike_detector;
    node_vect_.push_back(spike_detector_group);
  }
  else {
    throw ngpu_exception(std::string("Unknown neuron model name: ")
			 + model_name);
  }
  return NodeSeq(CreateNodeGroup(n_node, n_port), n_node);
}

