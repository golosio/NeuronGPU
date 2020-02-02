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
#include <cmath>
#include <stdlib.h>

#include "ngpu_exception.h"
#include "connect.h"

using namespace std;

int NetConnection::Connect(int i_source, int i_target, unsigned char port,
			   unsigned char syn_group, float weight, float delay) 
{
  if (delay<time_resolution_) {
    throw ngpu_exception("Delay must be >= time resolution");
  }
  
  int d_int = (int)round(delay/time_resolution_) - 1;
  TargetSyn tg = {i_target, port, syn_group, weight};
  Insert(d_int, i_source, tg);
  
  return 0;
}

int NetConnection::Insert(int d_int, int i_source, TargetSyn tg)
{
  int id;
  vector<ConnGroup> &conn = connection_[i_source];
  int conn_size = conn.size();
  for (id=0; id<conn_size && d_int>conn[id].delay; id++) {}
  if (id==conn_size || d_int!=conn[id].delay) {
    ConnGroup new_conn;
    new_conn.delay = d_int;
    new_conn.target_vect.push_back(tg);
    vector<ConnGroup>::iterator it = conn.begin() + id;
    conn.insert(it, new_conn);
  }
  else {
    conn[id].target_vect.push_back(tg);
  }

  return 0;
}

int NetConnection::Print()
{
  for (unsigned int i_source=0; i_source<connection_.size(); i_source++) {
    cout << "Source node: " << i_source << endl;
    ConnGroupPrint(i_source);
    cout << endl;
  }
  return 0;
}

int NetConnection::ConnGroupPrint(int i_source)
{
  vector<ConnGroup> &conn = connection_[i_source];
  for (unsigned int id=0; id<conn.size(); id++) {
    cout << "\tDelay: " << conn[id].delay << endl;
    std::vector<TargetSyn> tv = conn[id].target_vect;
    cout << "\tTargets: " << endl;
    for (unsigned int i=0; i<tv.size(); i++) {
      cout << "(" << tv[i].node << "," << (int)tv[i].port
	 << "," << (int)tv[i].syn_group
	   << "," << tv[i].weight << ")  ";
    }
    cout << endl;
  }
  
  return 0;
}

int NetConnection::MaxDelayNum()
{
  int max_delay_num = 0;
  for (unsigned int i_node=0; i_node<connection_.size(); i_node++) {
    vector<ConnGroup> &conn = connection_[i_node];
    int n_delays = conn.size();
    if (n_delays > max_delay_num) max_delay_num = n_delays;
  }

  return max_delay_num;
}

int NetConnection::NConnections()
{
  int n_conn = 0;
  for (unsigned int i_node=0; i_node<connection_.size(); i_node++) {
    vector<ConnGroup> &conn = connection_[i_node];
    for (unsigned int id=0; id<conn.size(); id++) {
      int n_target = conn.at(id).target_vect.size();
      n_conn += n_target;
    }
  }
  
  return n_conn;
}

template<>
int GetINode<int>(int i_node, int in)
{
  return i_node + in;
}

template<>
int GetINode<int*>(int* i_node, int in)
{
  return *(i_node + in);
}

ConnectionStatus NetConnection::GetConnectionStatus(ConnectionId conn_id)
{
  int i_source = conn_id.i_source_;
  int i_group = conn_id.i_group_;
  int i_conn = conn_id.i_conn_;
  vector<ConnGroup> &conn = connection_[i_source];
  std::vector<TargetSyn> tv = conn[i_group].target_vect;
  
  ConnectionStatus conn_stat;
  conn_stat.i_source = i_source;
  conn_stat.i_target = tv[i_conn].node;
  conn_stat.port = tv[i_conn].port;
  conn_stat.syn_group = tv[i_conn].syn_group;
  conn_stat.delay = time_resolution_*(conn[i_group].delay + 1);
  conn_stat.weight = tv[i_conn].weight;

  return conn_stat;
}

std::vector<ConnectionStatus> NetConnection::GetConnectionStatus
  (std::vector<ConnectionId> &conn_id_vect)
{
  std::vector<ConnectionStatus> conn_stat_vect;
  
  for (unsigned int i=0; i<conn_id_vect.size(); i++) {
    ConnectionId conn_id = conn_id_vect[i];
    ConnectionStatus conn_stat = GetConnectionStatus(conn_id);
    conn_stat_vect.push_back(conn_stat);
  }
  
  return conn_stat_vect;
}
