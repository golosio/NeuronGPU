/*
Copyright (C) 2019 Bruno Golosio
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

#ifndef NEURONGROUPH
#define NEURONGROUPH

#define MAX_N_NEURON_GROUPS 128

struct NeuronGroup
{
  int neuron_type_;
  int i_neuron_0_;
  int n_neurons_;
  int n_receptors_;
  double *get_spike_array_;
};

struct RK5DataStruct
{
  int neuron_type_;
  int i_neuron_0_;
};

#endif

