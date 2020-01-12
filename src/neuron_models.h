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

#ifndef NEURONMODELSH
#define NEURONMODELSH

enum NeuronModels {
  i_aeif_cond_beta_model = 0,
  N_NEURON_MODELS
};

const std::string neuron_model_name[N_NEURON_MODELS] = {
  "aeif_cond_beta"
};

#endif

