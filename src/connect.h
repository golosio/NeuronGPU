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

#ifndef CONNECTH
#define CONNECTH
#include <vector>
#include <algorithm>

extern float TimeResolution;

struct ConnectionId
{
  int i_source_;
  int i_group_;
  int i_conn_;
};

struct ConnectionStatus
{
  int i_source;
  int i_target;
  int i_port;
  int i_syn;
  float delay;
  float weight;
};

struct TargetSyn
{
  int node;
  unsigned char port;
  float weight;
};
  
struct ConnGroup // connections from the same source node with same delay
{
  int delay;
  std::vector<TargetSyn> target_vect;
};

template<class T>
int GetINode(T node, int in);

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
  
  int NConnections();

  ConnectionStatus GetConnectionStatus(ConnectionId conn_id);

  std::vector<ConnectionStatus> GetConnectionStatus(std::vector<ConnectionId>
						    &conn_id_vect);
  

  template<class T>
    std::vector<ConnectionId> GetConnections(T source, int n_source,
					     int i_target, int n_target,
					     int syn_type=0);

  template<class T>
    std::vector<ConnectionId> GetConnections(T source, int n_source,
					     int *i_target, int n_target,
					     int syn_type=0);

};

template<class T>
std::vector<ConnectionId> NetConnection::GetConnections(T source,
							int n_source,
							int i_target,
							int n_target,
							int /*syn_type*/)
{
  std::vector<ConnectionId> conn_id_vect;
  for (int is=0; is<n_source; is++) {
    int i_source = GetINode<T>(source, is);
    std::vector<ConnGroup> &conn = connection_[i_source];
    for (unsigned int id=0; id<conn.size(); id++) {
      std::vector<TargetSyn> tv = conn[id].target_vect;
      for (unsigned int i=0; i<tv.size(); i++) {
	int itg = tv[i].node;
	if ((itg>=i_target) && (itg<i_target+n_target)) {
	  ConnectionId conn_id;
	  conn_id.i_source_ = i_source;
	  conn_id.i_group_ = id;
	  conn_id.i_conn_ = i;
	  conn_id_vect.push_back(conn_id);
	}
      }
    }
  }
  
  return conn_id_vect;
}

template<class T>
std::vector<ConnectionId> NetConnection::GetConnections(T source,
							int n_source,
							int *i_target,
							int n_target,
							int /*syn_type*/)
{
  std::vector<int> target_vect(i_target, i_target+n_target);
  std::sort(target_vect.begin(), target_vect.end());
  
  std::vector<ConnectionId> conn_id_vect;
  for (int is=0; is<n_source; is++) {
    int i_source = GetINode<T>(source, is);
    std::vector<ConnGroup> &conn = connection_[i_source];
    for (unsigned int id=0; id<conn.size(); id++) {
      std::vector<TargetSyn> tv = conn[id].target_vect;
      for (unsigned int i=0; i<tv.size(); i++) {
	int itg = tv[i].node;
	// https://riptutorial.com/cplusplus/example/7270/using-a-sorted-vector-for-fast-element-lookup
	// check if itg is in target_vect
	std::vector<int>::iterator it = std::lower_bound(target_vect.begin(),
							 target_vect.end(),
							 itg);
	if (it != target_vect.end() && *it == itg) { // we found the element
	  ConnectionId conn_id;
	  conn_id.i_source_ = i_source;
	  conn_id.i_group_ = id;
	  conn_id.i_conn_ = i;
	  conn_id_vect.push_back(conn_id);
	}
      }
    }
  }
  
  return conn_id_vect;
}


#endif
