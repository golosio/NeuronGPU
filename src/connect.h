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

#ifndef CONNECTH
#define CONNECTH
#include <vector>

extern float TimeResolution;

struct TargetSyn
{
  int neuron;
  unsigned char port;
  float weight;
};
  
struct ConnGroup // connections from the same source neuron with same delay
{
  int delay;
  std::vector<TargetSyn> target_vect;
};

class NetConnection
{
 public:
  float time_resolution_;
  
  std::vector<std::vector<ConnGroup> > connection_;

  int Insert(int d_int, int i_source, TargetSyn tg);

  int Connect(int i_source, int i_target, unsigned char i_port, float weight,
	      float delay);

  int Print();
  
  int ConnGroupPrint(int i_source);

  int MaxDelayNum();

};

#endif
