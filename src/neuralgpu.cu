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
#include <iostream>
#include <string>
#include <algorithm>
#include <mpi.h>
#include <curand.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "send_spike.h"
#include "get_spike.h"
#include "connect_mpi.h"
#include "spike_mpi.h"
#include "spike_generator.h"
#include "multimeter.h"
#include "poisson.h"
#include "getRealTime.h"
#include "random.h"
#include "neuralgpu.h"
#include "nested_loop.h"
#include "dir_connect.h"

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

				    //using namespace std;

#define VERBOSE_TIME

NeuralGPU::NeuralGPU()
{
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  poiss_generator_ = new PoissonGenerator;
  multimeter_ = new Multimeter;
  net_connection_ = new NetConnection;
  connect_mpi_ = new ConnectMpi;

  SetRandomSeed(54321ULL);

  calibrate_flag_ = false;
  mpi_flag_ = false;
  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_poiss_nodes_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  connect_mpi_->net_connection_ = net_connection_;
  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;
  on_exception_ = ON_EXCEPTION_EXIT;

  NestedLoop::Init();
}

NeuralGPU::~NeuralGPU()
{
  delete poiss_generator_;
  delete multimeter_;
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    delete node_vect_[i];
  }
  delete net_connection_;
  delete connect_mpi_;
  curandDestroyGenerator(*random_generator_);
  delete random_generator_;
  if (calibrate_flag_) {
    FreeNodeGroupMap();
    FreeGetSpikeArrays();
  }

}

int NeuralGPU::SetRandomSeed(unsigned long long seed)
{
  kernel_seed_ = seed + 12345;
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(*random_generator_, seed));
  poiss_generator_->random_generator_ = random_generator_;

  return 0;
}

int NeuralGPU::SetTimeResolution(float time_res)
{
  time_resolution_ = time_res;
  net_connection_->time_resolution_ = time_res;
  
  return 0;
}

int NeuralGPU::SetMaxSpikeBufferSize(int max_size)
{
  max_spike_buffer_size_ = max_size;
  
  return 0;
}

int NeuralGPU::GetMaxSpikeBufferSize()
{
  return max_spike_buffer_size_;
}

int NeuralGPU::CreateNodeGroup(int n_nodes, int n_ports)
{
  int i_node_0 = node_group_map_.size();
  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    std::cerr << "Error: connect_mpi_.extern_connection_ and "
      "node_group_map_ must have the same size!\n";
  }
  if ((int)net_connection_->connection_.size() != i_node_0) {
    std::cerr << "Error: net_connection_.connection_ and "
      "node_group_map_ must have the same size!\n";
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert(node_group_map_.end(), n_nodes, i_group);
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_nodes, conn);

  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_nodes, conn_node);

  node_vect_[i_group]->Init(i_node_0, n_nodes, n_ports, i_group, &kernel_seed_);
  node_vect_[i_group]->get_spike_array_ = InitGetSpikeArray(n_nodes, n_ports);
  
  return i_node_0;
}

NodeSeq NeuralGPU::CreatePoissonGenerator(int n_nodes, float rate)
{
  CheckUncalibrated("Poisson generator cannot be created after calibration");
  if (n_poiss_nodes_ != 0) {
    throw ngpu_exception("Number of poisson generators cannot be modified.");
  }
  else if (n_nodes <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  
  n_poiss_nodes_ = n_nodes;               
 
  BaseNeuron *bn = new BaseNeuron;
  node_vect_.push_back(bn);
  int i_node_0 = CreateNodeGroup( n_nodes, 0);
  
  float lambda = rate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_nodes, lambda);
    
  return NodeSeq(i_node_0, n_nodes);
}


int NeuralGPU::CheckUncalibrated(std::string message)
{
  if (calibrate_flag_ == true) {
    throw ngpu_exception(message);
  }
  
  return 0;
}

int NeuralGPU::Calibrate()
{
  CheckUncalibrated("Calibration can be made only once");
  calibrate_flag_ = true;
  BuildDirectConnections();
  
  if (mpi_flag_) {
    std::cout << "Calibrating on host " << connect_mpi_->mpi_id_ << " ...\n";
  }
  else {
    std::cout << "Calibrating ...\n";
  }
  neural_time_ = t_min_;
  
  gpuErrchk(cudaMemcpyToSymbol(NeuralGPUMpiFlag, &mpi_flag_, sizeof(bool)));
	    
  NodeGroupArrayInit();
  
  max_spike_num_ = net_connection_->connection_.size()
    * net_connection_->MaxDelayNum();
  
  max_spike_per_host_ = net_connection_->connection_.size()
    * net_connection_->MaxDelayNum();

  SpikeInit(max_spike_num_);
  SpikeBufferInit(net_connection_, max_spike_buffer_size_);

  if (mpi_flag_) {
    // remove superfluous argument mpi_np
    connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				    max_spike_num_, connect_mpi_->mpi_np_,
				    max_spike_per_host_);
  }
  
  multimeter_->OpenFiles();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Calibrate(t_min_, time_resolution_);
  }
  //float x;
  //float y;
  //node_vect_[0].GetX(test_arr_idx, 1, &x);
  //node_vect_[0].GetY(test_var_idx, test_arr_idx, 1, &y);
  //fprintf(fp,"%f\t%f\n", x, y);

///////////////////////////////////

  return 0;
}

int NeuralGPU::Simulate(float sim_time)
{
  double SpikeBufferUpdate_time = 0;
  double poisson_generator_time = 0;
  double neuron_Update_time = 0;
  double copy_ext_spike_time = 0;
  double SendExternalSpike_time = 0;
  double SendSpikeToRemote_time = 0;
  double RecvSpikeFromRemote_time = 0;
  double NestedLoop_time = 0;
  double GetSpike_time = 0;
  double SpikeReset_time = 0;
  double ExternalSpikeReset_time = 0;
  double time_mark;

  sim_time_ = sim_time;
  
  if (!calibrate_flag_) {
    Calibrate();
  }
  multimeter_->WriteRecords(neural_time_);
  build_real_time_ = getRealTime();
  if (mpi_flag_) {
    std::cout << "Simulating on host " << connect_mpi_->mpi_id_ << " ...\n";
  }
  else {
    std::cout << "Simulating ...\n";
  }
  
  int Nt=(int)round(sim_time_/time_resolution_);
  printf("Neural activity simulation time: %.3f\n", sim_time_);
  float neur_t0 = neural_time_;
  for (int it=0; it<Nt; it++) {
    if (it%100==0) {
      printf("%.3f\n", neural_time_);
    }
    time_mark = getRealTime();
    SpikeBufferUpdate<<<(net_connection_->connection_.size()+1023)/1024,
      1024>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SpikeBufferUpdate_time += (getRealTime() - time_mark);
    time_mark = getRealTime();
    if (n_poiss_nodes_>0) {
      poiss_generator_->Update(Nt-it);
      poisson_generator_time += (getRealTime() - time_mark);
    }

    time_mark = getRealTime();
    neural_time_ = neur_t0 + time_resolution_*(it+1);
    for (unsigned int i=0; i<node_vect_.size(); i++) {
      node_vect_[i]->Update(it, neural_time_);
    }
    neuron_Update_time += (getRealTime() - time_mark);
    multimeter_->WriteRecords(neural_time_);
    if (mpi_flag_) {
      int n_ext_spike;
      time_mark = getRealTime();
      gpuErrchk(cudaMemcpy(&n_ext_spike, d_ExternalSpikeNum, sizeof(int),
			   cudaMemcpyDeviceToHost));
      copy_ext_spike_time += (getRealTime() - time_mark);

      if (n_ext_spike != 0) {
	//cout << "n_ext_spike " << n_ext_spike << endl;
	time_mark = getRealTime();
	SendExternalSpike<<<(n_ext_spike+1023)/1024, 1024>>>();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	SendExternalSpike_time += (getRealTime() - time_mark);
      }
      for (int ih=0; ih<connect_mpi_->mpi_np_; ih++) {
	if (ih == connect_mpi_->mpi_id_) {
	  time_mark = getRealTime();
	  connect_mpi_->SendSpikeToRemote(connect_mpi_->mpi_np_,
					  max_spike_per_host_);
	  SendSpikeToRemote_time += (getRealTime() - time_mark);
	}
	else {
	  time_mark = getRealTime();
	  connect_mpi_->RecvSpikeFromRemote(ih, max_spike_per_host_);
	  RecvSpikeFromRemote_time += (getRealTime() - time_mark);
	}
      }
    }

    int n_spikes;
    time_mark = getRealTime();
    gpuErrchk(cudaMemcpy(&n_spikes, d_SpikeNum, sizeof(int),
			 cudaMemcpyDeviceToHost));
    //cout << "n_spikes: " << n_spikes << endl;

    ClearGetSpikeArrays();    
    if (n_spikes > 0) {
      time_mark = getRealTime();
      NestedLoop::Run(n_spikes, d_SpikeTargetNum);
      NestedLoop_time += (getRealTime() - time_mark);
      time_mark = getRealTime();
    }
    for (unsigned int i=0; i<node_vect_.size(); i++) {
      if (node_vect_[i]->has_dir_conn_) {
	node_vect_[i]->SendDirectSpikes(neural_time_, time_resolution_/1000.0);
      }
    }

    for (unsigned int i=0; i<node_vect_.size(); i++) {
      if (node_vect_[i]->n_ports_>0) {
	GetSpikes<<<(node_vect_[i]->n_nodes_
		     *node_vect_[i]->n_ports_+1023)/1024, 1024>>>
	  (node_vect_[i]->i_group_, node_vect_[i]->n_nodes_,
	   node_vect_[i]->n_ports_,
	   node_vect_[i]->n_var_,
	   node_vect_[i]->port_weight_arr_,
	   node_vect_[i]->port_weight_arr_step_,
	   node_vect_[i]->port_weight_port_step_,
	   node_vect_[i]->port_input_arr_,
	   node_vect_[i]->port_input_arr_step_,
	   node_vect_[i]->port_input_port_step_);
	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
      }
    }
    GetSpike_time += (getRealTime() - time_mark);

    time_mark = getRealTime();
    SpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SpikeReset_time += (getRealTime() - time_mark);

    if (mpi_flag_) {
      time_mark = getRealTime();
      ExternalSpikeReset<<<1, 1>>>();
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      ExternalSpikeReset_time += (getRealTime() - time_mark);
    }
  }
  printf("%.3f\n", neural_time_);
  end_real_time_ = getRealTime();

  multimeter_->CloseFiles();
  //neuron.rk5.Free();

#ifdef VERBOSE_TIME
  std::cout << "\n";
  std::cout << "  SpikeBufferUpdate_time: " << SpikeBufferUpdate_time << "\n";
  std::cout << "  poisson_generator_time: " << poisson_generator_time << "\n";
  std::cout << "  neuron_Update_time: " << neuron_Update_time << "\n";
  std::cout << "  copy_ext_spike_time: " << copy_ext_spike_time << "\n";
  std::cout << "  SendExternalSpike_time: " << SendExternalSpike_time << "\n";
  std::cout << "  SendSpikeToRemote_time: " << SendSpikeToRemote_time << "\n";
  std::cout << "  RecvSpikeFromRemote_time: " << RecvSpikeFromRemote_time << "\n";
  std::cout << "  NestedLoop_time: " << NestedLoop_time << "\n";
  std::cout << "  GetSpike_time: " << GetSpike_time << "\n";
  std::cout << "  SpikeReset_time: " << SpikeReset_time << "\n";
  std::cout << "  ExternalSpikeReset_time: " << ExternalSpikeReset_time << "\n";
#endif
  printf("Build real time = %lf\n",
	 (build_real_time_ - start_real_time_));
  printf("Simulation real time = %lf\n",
	 (end_real_time_ - build_real_time_));

  return 0;
}

int NeuralGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int *i_port_arr,
			    int n_nodes)
{
  std::vector<BaseNeuron*> neur_vect;
  std::vector<int> i_neur_vect;
  std::vector<int> i_port_vect;
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_nodes; i++) {
    var_name_vect.push_back(var_name_arr[i]);
    int i_group = node_group_map_[i_node_arr[i]];
    i_neur_vect.push_back(i_node_arr[i] - node_vect_[i_group]->i_node_0_);
    i_port_vect.push_back(i_port_arr[i]);
    neur_vect.push_back(node_vect_[i_group]);
  }

  return multimeter_->CreateRecord(neur_vect, file_name, var_name_vect,
  				   i_neur_vect, i_port_vect);

}

int NeuralGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int n_nodes)
{
  std::vector<int> i_port_vect(n_nodes, 0);
  return CreateRecord(file_name, var_name_arr, i_node_arr,
		      i_port_vect.data(), n_nodes);
}

std::vector<std::vector<float>> *NeuralGPU::GetRecordData(int i_record)
{
  return multimeter_->GetRecordData(i_record);
}

int NeuralGPU::GetNodeSequenceOffset(int i_node, int n_nodes, int &i_group)
{
  if (i_node<0 || (i_node+n_nodes > (int)node_group_map_.size())) {
    throw ngpu_exception("Unrecognized node in getting node sequence offset");
  }
  i_group = node_group_map_[i_node];  
  if (node_group_map_[i_node+n_nodes-1] != i_group) {
    throw ngpu_exception("Nodes belong to different node groups "
			 "in setting parameter");
  }
  return node_vect_[i_group]->i_node_0_;
}
  
std::vector<int> NeuralGPU::GetNodeArrayWithOffset(int *i_node, int n_nodes,
						   int &i_group)
{
  int in0 = i_node[0];
  if (in0<0 || in0>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in setting parameter");
  }
  i_group = node_group_map_[in0];
  int i0 = node_vect_[i_group]->i_node_0_;
  std::vector<int> nodes;
  nodes.assign(i_node, i_node+n_nodes);
  for(int i=0; i<n_nodes; i++) {
    int in = nodes[i];
    if (in<0 || in>=(int)node_group_map_.size()) {
      throw ngpu_exception("Unrecognized node in setting parameter");
    }
    if (node_group_map_[in] != i_group) {
      throw ngpu_exception("Nodes belong to different node groups "
			   "in setting parameter");
    }
    nodes[i] -= i0;
  }
  return nodes;
}

int NeuralGPU::SetNeuronParam(int i_node, int n_nodes,
			      std::string param_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_nodes, i_group);
  
  return node_vect_[i_group]->SetScalParam(i_neuron, n_nodes, param_name, val);
}

int NeuralGPU::SetNeuronParam(int *i_node, int n_nodes,
			      std::string param_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_nodes,
						  i_group);
  return node_vect_[i_group]->SetScalParam(nodes.data(), n_nodes,
					   param_name, val);
}

int NeuralGPU::SetNeuronParam(int i_node, int n_nodes, std::string param_name,
			      float *params, int vect_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_nodes, i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {
      return node_vect_[i_group]->SetPortParam(i_neuron, n_nodes, param_name,
					       params, vect_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(i_neuron, n_nodes, param_name,
					      params, vect_size);
  }
}

int NeuralGPU::SetNeuronParam( int *i_node, int n_nodes,
			       std::string param_name, float *params,
			       int vect_size)
{
  int i_group;
  std::vector<int> node_vect = GetNodeArrayWithOffset(i_node, n_nodes,
						      i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->SetPortParam(node_vect.data(), n_nodes,
					     param_name, params, vect_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(node_vect.data(), n_nodes,
					      param_name, params, vect_size);
  }    
}

int NeuralGPU::IsNeuronScalParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalParam(param_name);
}

int NeuralGPU::IsNeuronPortParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortParam(param_name);
}

int NeuralGPU::IsNeuronArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayParam(param_name);
}

int NeuralGPU::SetNeuronVar(int i_node, int n_nodes,
			      std::string var_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_nodes, i_group);
  
  return node_vect_[i_group]->SetScalVar(i_neuron, n_nodes, var_name, val);
}

int NeuralGPU::SetNeuronVar(int *i_node, int n_nodes,
			      std::string var_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_nodes,
						  i_group);
  return node_vect_[i_group]->SetScalVar(nodes.data(), n_nodes,
					   var_name, val);
}

int NeuralGPU::SetNeuronVar(int i_node, int n_nodes, std::string var_name,
			      float *vars, int vect_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_nodes, i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {
      return node_vect_[i_group]->SetPortVar(i_neuron, n_nodes, var_name,
					       vars, vect_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(i_neuron, n_nodes, var_name,
					      vars, vect_size);
  }
}

int NeuralGPU::SetNeuronVar( int *i_node, int n_nodes,
			       std::string var_name, float *vars,
			       int vect_size)
{
  int i_group;
  std::vector<int> node_vect = GetNodeArrayWithOffset(i_node, n_nodes,
						      i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->SetPortVar(node_vect.data(), n_nodes,
					     var_name, vars, vect_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(node_vect.data(), n_nodes,
					      var_name, vars, vect_size);
  }    
}

int NeuralGPU::IsNeuronScalVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalVar(var_name);
}

int NeuralGPU::IsNeuronPortVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortVar(var_name);
}

int NeuralGPU::IsNeuronArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayVar(var_name);
}

int NeuralGPU::ConnectMpiInit(int argc, char *argv[])
{
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  int err = connect_mpi_->MpiInit(argc, argv);
  if (err==0) {
    mpi_flag_ = true;
  }
  
  return err;
}

int NeuralGPU::MpiId()
{
  return connect_mpi_->mpi_id_;
}

int NeuralGPU::MpiNp()
{
  return connect_mpi_->mpi_np_;
}

int NeuralGPU::ProcMaster()
{
  return connect_mpi_->ProcMaster();
}

int NeuralGPU::MpiFinalize()
{
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  
  return 0;
}

unsigned int *NeuralGPU::RandomInt(size_t n)
{
  return curand_int(*random_generator_, n);
}

float *NeuralGPU::RandomUniform(size_t n)
{
  return curand_uniform(*random_generator_, n);
}

float *NeuralGPU::RandomNormal(size_t n, float mean, float stddev)
{
  return curand_normal(*random_generator_, n, mean, stddev);
}

float *NeuralGPU::RandomNormalClipped(size_t n, float mean, float stddev,
				      float vmin, float vmax)
{
  int n_extra = n/10;
  if (n_extra<1024) {
    n_extra=1024;
  }
  int i_extra = 0;
  float *arr = curand_normal(*random_generator_, n, mean, stddev);
  float *arr_extra;
  for (size_t i=0; i<n; i++) {
    while (arr[i]<vmin || arr[i]>vmax) {
      if (i_extra==0) {
	arr_extra = curand_normal(*random_generator_, n_extra, mean, stddev);
      }
      arr[i] = arr_extra[i_extra];
      i_extra++;
      if (i_extra==n_extra) {
	i_extra = 0;
	delete[](arr_extra);
      }
    }
  }
  if (i_extra != 0) {
    delete[](arr_extra);
  }
  return arr; 
}

int NeuralGPU::BuildDirectConnections()
{
  for (unsigned int iv=0; iv<node_vect_.size(); iv++) {
    if (node_vect_[iv]->has_dir_conn_==true) {
      std::vector<DirectConnection> dir_conn_vect;
      int i0 = node_vect_[iv]->i_node_0_;
      int n = node_vect_[iv]->n_nodes_;
      for (int i_source=i0; i_source<i0+n; i_source++) {
	vector<ConnGroup> &conn = net_connection_->connection_[i_source];
	for (unsigned int id=0; id<conn.size(); id++) {
	  std::vector<TargetSyn> tv = conn[id].target_vect;
	  for (unsigned int i=0; i<tv.size(); i++) {
	    DirectConnection dir_conn;
	    dir_conn.irel_source_ = i_source - i0;
	    dir_conn.i_target_ = tv[i].node;
	    dir_conn.port_ = tv[i].port;
	    dir_conn.weight_ = tv[i].weight;
	    dir_conn.delay_ = time_resolution_*(conn[id].delay+1);
	    dir_conn_vect.push_back(dir_conn);
	  }
	}
      }
      long n_dir_conn = dir_conn_vect.size();
      node_vect_[iv]->n_dir_conn_ = n_dir_conn;
      
      DirectConnection *d_dir_conn_array;
      gpuErrchk(cudaMalloc(&d_dir_conn_array,
			   n_dir_conn*sizeof(DirectConnection )));
      gpuErrchk(cudaMemcpy(d_dir_conn_array, dir_conn_vect.data(),
			   n_dir_conn*sizeof(DirectConnection),
			   cudaMemcpyHostToDevice));
      node_vect_[iv]->d_dir_conn_array_ = d_dir_conn_array;
    }
  }

  return 0;
}
