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
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <curand.h>
#include "spike_buffer.h"
#include "cuda_error.h"
#include "send_spike.h"
#include "get_spike.h"
#include "connect_mpi.h"

#include "spike_generator.h"
#include "multimeter.h"
#include "poisson.h"
#include "getRealTime.h"
#include "random.h"
#include "neurongpu.h"
#include "nested_loop.h"
#include "dir_connect.h"
#include "rev_spike.h"

#ifdef HAVE_MPI
#include <mpi.h>
#include "spike_mpi.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

				    //#define VERBOSE_TIME

__constant__ float NeuronGPUTime;
__constant__ int NeuronGPUTimeIdx;
__constant__ float NeuronGPUTimeResolution;

NeuronGPU::NeuronGPU()
{
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  poiss_generator_ = new PoissonGenerator;
  multimeter_ = new Multimeter;
  net_connection_ = new NetConnection;
  
  SetRandomSeed(54321ULL);
  
  calibrate_flag_ = false;

  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 20;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_poiss_node_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms

  error_flag_ = false;
  error_message_ = "";
  error_code_ = 0;
  
  on_exception_ = ON_EXCEPTION_EXIT;

  verbosity_level_ = 3;
  
#ifdef HAVE_MPI
  connect_mpi_ = new ConnectMpi;
  mpi_flag_ = false;
  connect_mpi_->net_connection_ = net_connection_;
#endif
  
  NestedLoop::Init();

  SpikeBufferUpdate_time_ = 0;
  poisson_generator_time_ = 0;
  neuron_Update_time_ = 0;
  copy_ext_spike_time_ = 0;
  SendExternalSpike_time_ = 0;
  SendSpikeToRemote_time_ = 0;
  RecvSpikeFromRemote_time_ = 0;
  NestedLoop_time_ = 0;
  GetSpike_time_ = 0;
  SpikeReset_time_ = 0;
  ExternalSpikeReset_time_ = 0;
  first_simulation_flag_ = true;
}

NeuronGPU::~NeuronGPU()
{
  multimeter_->CloseFiles();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  if (calibrate_flag_) {
    FreeNodeGroupMap();
    FreeGetSpikeArrays();
  }

  for (unsigned int i=0; i<node_vect_.size(); i++) {
    delete node_vect_[i];
  }

#ifdef HAVE_MPI
  delete connect_mpi_;
#endif

  delete net_connection_;
  delete multimeter_;
  delete poiss_generator_;
  curandDestroyGenerator(*random_generator_);
  delete random_generator_;
}

int NeuronGPU::SetRandomSeed(unsigned long long seed)
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

int NeuronGPU::SetTimeResolution(float time_res)
{
  time_resolution_ = time_res;
  net_connection_->time_resolution_ = time_res;
  
  return 0;
}

int NeuronGPU::SetMaxSpikeBufferSize(int max_size)
{
  max_spike_buffer_size_ = max_size;
  
  return 0;
}

int NeuronGPU::GetMaxSpikeBufferSize()
{
  return max_spike_buffer_size_;
}

int NeuronGPU::CreateNodeGroup(int n_node, int n_port)
{
  int i_node_0 = node_group_map_.size();

#ifdef HAVE_MPI
  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    throw ngpu_exception("Error: connect_mpi_.extern_connection_ and "
			 "node_group_map_ must have the same size!");
  }
#endif

  if ((int)net_connection_->connection_.size() != i_node_0) {
    throw ngpu_exception("Error: net_connection_.connection_ and "
			 "node_group_map_ must have the same size!");
  }
  if ((net_connection_->connection_.size() + n_node) > MAX_N_NEURON) {
    throw ngpu_exception(std::string("Maximum number of neurons ")
			 + std::to_string(MAX_N_NEURON) + " exceeded");
  }
  if (n_port > MAX_N_PORT) {
    throw ngpu_exception(std::string("Maximum number of ports ")
			 + std::to_string(MAX_N_PORT) + " exceeded");
  }
  int i_group = node_vect_.size() - 1;
  node_group_map_.insert(node_group_map_.end(), n_node, i_group);
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_node, conn);

#ifdef HAVE_MPI
  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_node, conn_node);
#endif
  
  node_vect_[i_group]->Init(i_node_0, n_node, n_port, i_group, &kernel_seed_);
  node_vect_[i_group]->get_spike_array_ = InitGetSpikeArray(n_node, n_port);
  
  return i_node_0;
}

NodeSeq NeuronGPU::CreatePoissonGenerator(int n_node, float rate)
{
  CheckUncalibrated("Poisson generator cannot be created after calibration");
  if (n_poiss_node_ != 0) {
    throw ngpu_exception("Number of poisson generators cannot be modified.");
  }
  else if (n_node <= 0) {
    throw ngpu_exception("Number of nodes must be greater than zero.");
  }
  
  n_poiss_node_ = n_node;               
 
  BaseNeuron *bn = new BaseNeuron;
  node_vect_.push_back(bn);
  int i_node_0 = CreateNodeGroup( n_node, 0);
  
  float lambda = rate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_node, lambda);
    
  return NodeSeq(i_node_0, n_node);
}


int NeuronGPU::CheckUncalibrated(std::string message)
{
  if (calibrate_flag_ == true) {
    throw ngpu_exception(message);
  }
  
  return 0;
}

int NeuronGPU::Calibrate()
{
  CheckUncalibrated("Calibration can be made only once");
  calibrate_flag_ = true;
  BuildDirectConnections();

  if (verbosity_level_>=1) {
#ifdef HAVE_MPI
    if (mpi_flag_) {
      std::cout << "Calibrating on host " << connect_mpi_->mpi_id_ << " ...\n";
    }
    else {
      std::cout << "Calibrating ...\n";
    }
    gpuErrchk(cudaMemcpyToSymbol(NeuronGPUMpiFlag, &mpi_flag_, sizeof(bool)));
#else
    std::cout << "Calibrating ...\n";
#endif
  }
  
  neural_time_ = t_min_;
  	    
  NodeGroupArrayInit();
  
  max_spike_num_ = net_connection_->connection_.size()
    * net_connection_->MaxDelayNum();
  
  max_spike_per_host_ = net_connection_->connection_.size()
    * net_connection_->MaxDelayNum();

  SpikeInit(max_spike_num_);
  SpikeBufferInit(net_connection_, max_spike_buffer_size_);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    // remove superfluous argument mpi_np
    connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				    max_spike_num_, connect_mpi_->mpi_np_,
				    max_spike_per_host_);
  }
#endif
  
  if (net_connection_->NRevConnections()>0) {
    RevSpikeInit(net_connection_, round(t_min_/time_resolution_)); 
  }
  
  multimeter_->OpenFiles();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Calibrate(t_min_, time_resolution_);
  }
  
  SynGroupCalibrate();
  
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTimeResolution, &time_resolution_,
			       sizeof(float)));
///////////////////////////////////

  return 0;
}

int NeuronGPU::Simulate(float sim_time) {
  sim_time_ = sim_time;
  return Simulate();
}

int NeuronGPU::Simulate()
{
  StartSimulation();
  
  for (int it=0; it<Nt_; it++) {
    if (it%100==0 && verbosity_level_>=2) {
      printf("%.3f\n", neural_time_);
    }
    SimulationStep();
  }
  EndSimulation();

  return 0;
}

int NeuronGPU::StartSimulation()
{
  if (!calibrate_flag_) {
    Calibrate();
  }
  if (first_simulation_flag_) {
    gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTime, &neural_time_, sizeof(float)));
    multimeter_->WriteRecords(neural_time_);
    build_real_time_ = getRealTime();
    first_simulation_flag_ = false;
  }

  if (verbosity_level_>=1) {
#ifdef HAVE_MPI
    if (mpi_flag_) {
      std::cout << "Simulating on host " << connect_mpi_->mpi_id_ << " ...\n";
    }
    else {
      std::cout << "Simulating ...\n";
    }
#else
    std::cout << "Simulating ...\n";
#endif
    printf("Neural activity simulation time: %.3f\n", sim_time_);
  }
  
  neur_t0_ = neural_time_;
  it_ = 0;
  Nt_ = (int)round(sim_time_/time_resolution_);
  
  return 0;
}

int NeuronGPU::EndSimulation()
{
  if (verbosity_level_>=2) {
    printf("%.3f\n", neural_time_);
  }
  end_real_time_ = getRealTime();

  //multimeter_->CloseFiles();
  //neuron.rk5.Free();

  if (verbosity_level_>=3) {
    std::cout << "\n";
    std::cout << "  SpikeBufferUpdate_time: " << SpikeBufferUpdate_time_
	      << "\n";
    std::cout << "  poisson_generator_time: " << poisson_generator_time_
	      << "\n";
    std::cout << "  neuron_Update_time: " << neuron_Update_time_ << "\n";
    std::cout << "  copy_ext_spike_time: " << copy_ext_spike_time_ << "\n";
    std::cout << "  SendExternalSpike_time: " << SendExternalSpike_time_
	      << "\n";
    std::cout << "  SendSpikeToRemote_time: " << SendSpikeToRemote_time_
	      << "\n";
    std::cout << "  RecvSpikeFromRemote_time: " << RecvSpikeFromRemote_time_
	      << "\n";
    std::cout << "  NestedLoop_time: " << NestedLoop_time_ << "\n";
    std::cout << "  GetSpike_time: " << GetSpike_time_ << "\n";
    std::cout << "  SpikeReset_time: " << SpikeReset_time_ << "\n";
    std::cout << "  ExternalSpikeReset_time: " << ExternalSpikeReset_time_
	      << "\n";
  }
  if (verbosity_level_>=1) {
    printf("Build real time = %lf\n",
	   (build_real_time_ - start_real_time_));
    printf("Simulation real time = %lf\n",
	   (end_real_time_ - build_real_time_));
  }
  
  return 0;
}


int NeuronGPU::SimulationStep()
{
  if (first_simulation_flag_) {
    StartSimulation();
  }
  double time_mark;

  time_mark = getRealTime();
  SpikeBufferUpdate<<<(net_connection_->connection_.size()+1023)/1024,
    1024>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  SpikeBufferUpdate_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();
  if (n_poiss_node_>0) {
    poiss_generator_->Update(Nt_-it_);
    poisson_generator_time_ += (getRealTime() - time_mark);
  }
  time_mark = getRealTime();
  neural_time_ = neur_t0_ + time_resolution_*(it_+1);
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTime, &neural_time_, sizeof(float)));
  int time_idx = (int)round(neur_t0_/time_resolution_) + it_ + 1;
  gpuErrchk(cudaMemcpyToSymbol(NeuronGPUTimeIdx, &time_idx, sizeof(int)));

  if (ConnectionSpikeTimeFlag) {
    if ( (time_idx & 0xffff) == 0x8000) {
      ResetConnectionSpikeTimeUp(net_connection_);
    }
    else if ( (time_idx & 0xffff) == 0) {
      ResetConnectionSpikeTimeDown(net_connection_);
    }
  }
    
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    node_vect_[i]->Update(it_, neural_time_);
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  
  neuron_Update_time_ += (getRealTime() - time_mark);
  multimeter_->WriteRecords(neural_time_);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    int n_ext_spike;
    time_mark = getRealTime();
    gpuErrchk(cudaMemcpy(&n_ext_spike, d_ExternalSpikeNum, sizeof(int),
			 cudaMemcpyDeviceToHost));
    copy_ext_spike_time_ += (getRealTime() - time_mark);

    if (n_ext_spike != 0) {
      time_mark = getRealTime();
      SendExternalSpike<<<(n_ext_spike+1023)/1024, 1024>>>();
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      SendExternalSpike_time_ += (getRealTime() - time_mark);
    }
    for (int ih=0; ih<connect_mpi_->mpi_np_; ih++) {
      if (ih == connect_mpi_->mpi_id_) {
	time_mark = getRealTime();
	connect_mpi_->SendSpikeToRemote(connect_mpi_->mpi_np_,
					max_spike_per_host_);
	SendSpikeToRemote_time_ += (getRealTime() - time_mark);
      }
      else {
	time_mark = getRealTime();
	connect_mpi_->RecvSpikeFromRemote(ih, max_spike_per_host_);
	RecvSpikeFromRemote_time_ += (getRealTime() - time_mark);
      }
    }
  }
#endif
    
  int n_spikes;
  time_mark = getRealTime();
  gpuErrchk(cudaMemcpy(&n_spikes, d_SpikeNum, sizeof(int),
		       cudaMemcpyDeviceToHost));

  ClearGetSpikeArrays();    
  if (n_spikes > 0) {
    time_mark = getRealTime();
    NestedLoop::Run(n_spikes, d_SpikeTargetNum, 0);
    NestedLoop_time_ += (getRealTime() - time_mark);
  }
  time_mark = getRealTime();
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->has_dir_conn_) {
      node_vect_[i]->SendDirectSpikes(neural_time_, time_resolution_/1000.0);
    }
  }
  poisson_generator_time_ += (getRealTime() - time_mark);
  time_mark = getRealTime();
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->n_port_>0) {

      int grid_dim_x = (node_vect_[i]->n_node_+1023)/1024;
      int grid_dim_y = node_vect_[i]->n_port_;
      dim3 grid_dim(grid_dim_x, grid_dim_y);
      //dim3 block_dim(1024,1);
					    
      GetSpikes<<<grid_dim, 1024>>> //block_dim>>>
	(node_vect_[i]->get_spike_array_, node_vect_[i]->n_node_,
	 node_vect_[i]->n_port_,
	 node_vect_[i]->n_var_,
	 node_vect_[i]->port_weight_arr_,
	 node_vect_[i]->port_weight_arr_step_,
	 node_vect_[i]->port_weight_port_step_,
	 node_vect_[i]->port_input_arr_,
	 node_vect_[i]->port_input_arr_step_,
	 node_vect_[i]->port_input_port_step_);
      // gpuErrchk( cudaPeekAtLastError() );
      // gpuErrchk( cudaDeviceSynchronize() );
    }
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  GetSpike_time_ += (getRealTime() - time_mark);

  time_mark = getRealTime();
  SpikeReset<<<1, 1>>>();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  SpikeReset_time_ += (getRealTime() - time_mark);

#ifdef HAVE_MPI
  if (mpi_flag_) {
    time_mark = getRealTime();
    ExternalSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    ExternalSpikeReset_time_ += (getRealTime() - time_mark);
  }
#endif

  if (net_connection_->NRevConnections()>0) {
    //time_mark = getRealTime();
    RevSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    RevSpikeBufferUpdate<<<(net_connection_->connection_.size()+1023)/1024,
      1024>>>(net_connection_->connection_.size());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    unsigned int n_rev_spikes;
    gpuErrchk(cudaMemcpy(&n_rev_spikes, d_RevSpikeNum, sizeof(unsigned int),
			 cudaMemcpyDeviceToHost));
    if (n_rev_spikes > 0) {
      NestedLoop::Run(n_rev_spikes, d_RevSpikeNConn, 1);
    }      
    //RevSpikeBufferUpdate_time_ += (getRealTime() - time_mark);
  }
  it_++;
  
  return 0;
}

int NeuronGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int *port_arr,
			    int n_node)
{
  std::vector<BaseNeuron*> neur_vect;
  std::vector<int> i_neur_vect;
  std::vector<int> port_vect;
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_node; i++) {
    var_name_vect.push_back(var_name_arr[i]);
    int i_group = node_group_map_[i_node_arr[i]];
    i_neur_vect.push_back(i_node_arr[i] - node_vect_[i_group]->i_node_0_);
    port_vect.push_back(port_arr[i]);
    neur_vect.push_back(node_vect_[i_group]);
  }

  return multimeter_->CreateRecord(neur_vect, file_name, var_name_vect,
  				   i_neur_vect, port_vect);

}

int NeuronGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_node_arr, int n_node)
{
  std::vector<int> port_vect(n_node, 0);
  return CreateRecord(file_name, var_name_arr, i_node_arr,
		      port_vect.data(), n_node);
}

std::vector<std::vector<float> > *NeuronGPU::GetRecordData(int i_record)
{
  return multimeter_->GetRecordData(i_record);
}

int NeuronGPU::GetNodeSequenceOffset(int i_node, int n_node, int &i_group)
{
  if (i_node<0 || (i_node+n_node > (int)node_group_map_.size())) {
    throw ngpu_exception("Unrecognized node in getting node sequence offset");
  }
  i_group = node_group_map_[i_node];  
  if (node_group_map_[i_node+n_node-1] != i_group) {
    throw ngpu_exception("Nodes belong to different node groups "
			 "in setting parameter");
  }
  return node_vect_[i_group]->i_node_0_;
}
  
std::vector<int> NeuronGPU::GetNodeArrayWithOffset(int *i_node, int n_node,
						   int &i_group)
{
  int in0 = i_node[0];
  if (in0<0 || in0>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in setting parameter");
  }
  i_group = node_group_map_[in0];
  int i0 = node_vect_[i_group]->i_node_0_;
  std::vector<int> nodes;
  nodes.assign(i_node, i_node+n_node);
  for(int i=0; i<n_node; i++) {
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

int NeuronGPU::SetNeuronParam(int i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalParam(i_neuron, n_node, param_name, val);
}

int NeuronGPU::SetNeuronParam(int *i_node, int n_node,
			      std::string param_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalParam(nodes.data(), n_node,
					   param_name, val);
}

int NeuronGPU::SetNeuronParam(int i_node, int n_node, std::string param_name,
			      float *param, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {
      return node_vect_[i_group]->SetPortParam(i_neuron, n_node, param_name,
					       param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(i_neuron, n_node, param_name,
					      param, array_size);
  }
}

int NeuronGPU::SetNeuronParam( int *i_node, int n_node,
			       std::string param_name, float *param,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->SetPortParam(nodes.data(), n_node,
					     param_name, param, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayParam(nodes.data(), n_node,
					      param_name, param, array_size);
  }    
}

int NeuronGPU::IsNeuronScalParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalParam(param_name);
}

int NeuronGPU::IsNeuronPortParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortParam(param_name);
}

int NeuronGPU::IsNeuronArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayParam(param_name);
}

int NeuronGPU::SetNeuronIntVar(int i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetIntVar(i_neuron, n_node, var_name, val);
}

int NeuronGPU::SetNeuronIntVar(int *i_node, int n_node,
			      std::string var_name, int val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetIntVar(nodes.data(), n_node,
					var_name, val);
}

int NeuronGPU::SetNeuronVar(int i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  
  return node_vect_[i_group]->SetScalVar(i_neuron, n_node, var_name, val);
}

int NeuronGPU::SetNeuronVar(int *i_node, int n_node,
			      std::string var_name, float val)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  return node_vect_[i_group]->SetScalVar(nodes.data(), n_node,
					   var_name, val);
}

int NeuronGPU::SetNeuronVar(int i_node, int n_node, std::string var_name,
			      float *var, int array_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {
      return node_vect_[i_group]->SetPortVar(i_neuron, n_node, var_name,
					       var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(i_neuron, n_node, var_name,
					      var, array_size);
  }
}

int NeuronGPU::SetNeuronVar( int *i_node, int n_node,
			       std::string var_name, float *var,
			       int array_size)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->SetPortVar(nodes.data(), n_node,
					   var_name, var, array_size);
  }
  else {
    return node_vect_[i_group]->SetArrayVar(nodes.data(), n_node,
					    var_name, var, array_size);
  }    
}

int NeuronGPU::IsNeuronIntVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsIntVar(var_name);
}

int NeuronGPU::IsNeuronScalVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsScalVar(var_name);
}

int NeuronGPU::IsNeuronPortVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsPortVar(var_name);
}

int NeuronGPU::IsNeuronArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  
  return node_vect_[i_group]->IsArrayVar(var_name);
}


int NeuronGPU::GetNeuronParamSize(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayParam(param_name)!=0) {
    return node_vect_[i_group]->GetArrayParamSize(i_neuron, param_name);
  }
  else {
    return node_vect_[i_group]->GetParamSize(param_name);
  }
}

int NeuronGPU::GetNeuronVarSize(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  if (node_vect_[i_group]->IsArrayVar(var_name)!=0) {
    return node_vect_[i_group]->GetArrayVarSize(i_neuron, var_name);
  }
  else {
    return node_vect_[i_group]->GetVarSize(var_name);
  }
}


float *NeuronGPU::GetNeuronParam(int i_node, int n_node,
				 std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {
    return node_vect_[i_group]->GetPortParam(i_neuron, n_node, param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NeuronGPU::GetNeuronParam( int *i_node, int n_node,
				  std::string param_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalParam(param_name)) {
    return node_vect_[i_group]->GetScalParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsPortParam(param_name)) {  
    return node_vect_[i_group]->GetPortParam(nodes.data(), n_node,
					     param_name);
  }
  else if (node_vect_[i_group]->IsArrayParam(param_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array parameters for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayParam(nodes[0], param_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized parameter ")
			 + param_name);
  }
}

float *NeuronGPU::GetArrayParam(int i_node, std::string param_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayParam(i_neuron, param_name);
}

int *NeuronGPU::GetNeuronIntVar(int i_node, int n_node,
				std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(i_neuron, n_node, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized integer variable ")
			 + var_name);
  }
}

int *NeuronGPU::GetNeuronIntVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsIntVar(var_name)) {
    return node_vect_[i_group]->GetIntVar(nodes.data(), n_node,
					     var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetNeuronVar(int i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_node, i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {
    return node_vect_[i_group]->GetPortVar(i_neuron, n_node, var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetNeuronVar(int *i_node, int n_node,
			       std::string var_name)
{
  int i_group;
  std::vector<int> nodes = GetNodeArrayWithOffset(i_node, n_node,
						  i_group);
  if (node_vect_[i_group]->IsScalVar(var_name)) {
    return node_vect_[i_group]->GetScalVar(nodes.data(), n_node,
					     var_name);
  }
  else if (node_vect_[i_group]->IsPortVar(var_name)) {  
    return node_vect_[i_group]->GetPortVar(nodes.data(), n_node,
					   var_name);
  }
  else if (node_vect_[i_group]->IsArrayVar(var_name)) {
    if (n_node != 1) {
      throw ngpu_exception("Cannot get array variables for more than one node"
			   "at a time");
    }
    return node_vect_[i_group]->GetArrayVar(nodes[0], var_name);
  }
  else {
    throw ngpu_exception(std::string("Unrecognized variable ")
			 + var_name);
  }
}

float *NeuronGPU::GetArrayVar(int i_node, std::string var_name)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetArrayVar(i_neuron, var_name);
}

int NeuronGPU::ConnectMpiInit(int argc, char *argv[])
{
#ifdef HAVE_MPI
  CheckUncalibrated("MPI connections cannot be initialized after calibration");
  int err = connect_mpi_->MpiInit(argc, argv);
  if (err==0) {
    mpi_flag_ = true;
  }
  
  return err;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NeuronGPU::MpiId()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_id_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

int NeuronGPU::MpiNp()
{
#ifdef HAVE_MPI
  return connect_mpi_->mpi_np_;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif

}

int NeuronGPU::ProcMaster()
{
#ifdef HAVE_MPI
  return connect_mpi_->ProcMaster();
#else
  throw ngpu_exception("MPI is not available in your build");
#endif  
}

int NeuronGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if (mpi_flag_) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }
  
  return 0;
#else
  throw ngpu_exception("MPI is not available in your build");
#endif
}

unsigned int *NeuronGPU::RandomInt(size_t n)
{
  return curand_int(*random_generator_, n);
}

float *NeuronGPU::RandomUniform(size_t n)
{
  return curand_uniform(*random_generator_, n);
}

float *NeuronGPU::RandomNormal(size_t n, float mean, float stddev)
{
  return curand_normal(*random_generator_, n, mean, stddev);
}

float *NeuronGPU::RandomNormalClipped(size_t n, float mean, float stddev,
				      float vmin, float vmax)
{
  int n_extra = n/10;
  if (n_extra<1024) {
    n_extra=1024;
  }
  int i_extra = 0;
  float *arr = curand_normal(*random_generator_, n, mean, stddev);
  float *arr_extra = NULL;
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
  if (arr_extra != NULL) {
    delete[](arr_extra);
  }
  return arr; 
}

int NeuronGPU::BuildDirectConnections()
{
  for (unsigned int iv=0; iv<node_vect_.size(); iv++) {
    if (node_vect_[iv]->has_dir_conn_) {
      std::vector<DirectConnection> dir_conn_vect;
      int i0 = node_vect_[iv]->i_node_0_;
      int n = node_vect_[iv]->n_node_;
      for (int i_source=i0; i_source<i0+n; i_source++) {
	std::vector<ConnGroup> &conn = net_connection_->connection_[i_source];
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
      uint64_t n_dir_conn = dir_conn_vect.size();
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

std::vector<std::string> NeuronGPU::GetIntVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetIntVarNames();
}

std::vector<std::string> NeuronGPU::GetScalVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalVarNames();
}

int NeuronGPU::GetNIntVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNIntVar();
}

int NeuronGPU::GetNScalVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalVar();
}

std::vector<std::string> NeuronGPU::GetPortVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortVarNames();
}

int NeuronGPU::GetNPortVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortVar();
}


std::vector<std::string> NeuronGPU::GetScalParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetScalParamNames();
}

int NeuronGPU::GetNScalParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNScalParam();
}

std::vector<std::string> NeuronGPU::GetPortParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetPortParamNames();
}

int NeuronGPU::GetNPortParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNPortParam();
}


std::vector<std::string> NeuronGPU::GetArrayParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayParamNames();
}

int NeuronGPU::GetNArrayParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayParam();
}


std::vector<std::string> NeuronGPU::GetArrayVarNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading array variable names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetArrayVarNames();
}

int NeuronGPU::GetNArrayVar(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of array "
			 "variables");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNArrayVar();
}

ConnectionStatus NeuronGPU::GetConnectionStatus(ConnectionId conn_id) {
  ConnectionStatus conn_stat = net_connection_->GetConnectionStatus(conn_id);
  if (calibrate_flag_ == true) {
    int i_source = conn_id.i_source_;
    int i_group = conn_id.i_group_;
    int i_conn = conn_id.i_conn_;
    int n_spike_buffer = net_connection_->connection_.size();
    conn_stat.weight = 0;
    float *d_weight_pt
      = h_ConnectionGroupTargetWeight[i_group*n_spike_buffer+i_source] + i_conn;
    gpuErrchk(cudaMemcpy(&conn_stat.weight, d_weight_pt, sizeof(float),
			 cudaMemcpyDeviceToHost));
  }
  return conn_stat;
}

std::vector<ConnectionStatus> NeuronGPU::GetConnectionStatus(std::vector
							     <ConnectionId>
							     &conn_id_vect) {
  std::vector<ConnectionStatus> conn_stat_vect;
  for (unsigned int i=0; i<conn_id_vect.size(); i++) {
    ConnectionStatus conn_stat = GetConnectionStatus(conn_id_vect[i]);
    conn_stat_vect.push_back(conn_stat);
  }
  return conn_stat_vect;
}
  
std::vector<ConnectionId> NeuronGPU::GetConnections(int i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }

  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NeuronGPU::GetConnections(int *i_source, int n_source,
						    int i_target, int n_target,
						    int syn_group) {
  if (n_target<=0) {
    i_target = 0;
    n_target = net_connection_->connection_.size();
  }
    
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NeuronGPU::GetConnections(int i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  if (n_source<=0) {
    i_source = 0;
    n_source = net_connection_->connection_.size();
  }
  
  return net_connection_->GetConnections<int>(i_source, n_source, i_target,
					      n_target, syn_group);    
}

std::vector<ConnectionId> NeuronGPU::GetConnections(int *i_source, int n_source,
						    int *i_target, int n_target,
						    int syn_group) {
  
  return net_connection_->GetConnections<int*>(i_source, n_source, i_target,
					       n_target, syn_group);
  
}


std::vector<ConnectionId> NeuronGPU::GetConnections(NodeSeq source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n, target.i0,
					      target.n, syn_group);
}

std::vector<ConnectionId> NeuronGPU::GetConnections(std::vector<int> source,
						    NodeSeq target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.i0, target.n,
					       syn_group);
}


std::vector<ConnectionId> NeuronGPU::GetConnections(NodeSeq source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int>(source.i0, source.n,
					      target.data(), target.size(),
					      syn_group);
}

std::vector<ConnectionId> NeuronGPU::GetConnections(std::vector<int> source,
						    std::vector<int> target,
						    int syn_group) {
  return net_connection_->GetConnections<int*>(source.data(), source.size(),
					       target.data(), target.size(),
					       syn_group);
}

int NeuronGPU::ActivateSpikeCount(int i_node, int n_node)
{
  CheckUncalibrated("Spike count must be activated before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateSpikeCount();

  return 0;
}

int NeuronGPU::ActivateRecSpikeTimes(int i_node, int n_node,
				     int max_n_rec_spike_times)
{
  CheckUncalibrated("Spike time recording must be activated "
		    "before calibration");
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception("Spike count must be activated for all and only "
			 " the nodes of the same group");
  }
  node_vect_[i_group]->ActivateRecSpikeTimes(max_n_rec_spike_times);

  return 0;
}

int NeuronGPU::GetNRecSpikeTimes(int i_node)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  return node_vect_[i_group]->GetNRecSpikeTimes(i_neuron);
}

std::vector<float> NeuronGPU::GetRecSpikeTimes(int i_node)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, 1, i_group);
  return node_vect_[i_group]->GetRecSpikeTimes(i_neuron);
}

int NeuronGPU::PushSpikesToNodes(int n_spikes, int *node_id,
				 float *spike_height)
{
  int *d_node_id;
  float *d_spike_height;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_spike_height, n_spikes*sizeof(float)));
  gpuErrchk(cudaMemcpy(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_spike_height, spike_height, n_spikes*sizeof(float),
		       cudaMemcpyHostToDevice));
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id,
						     d_spike_height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));
  gpuErrchk(cudaFree(d_spike_height));

  return 0;
}

int NeuronGPU::PushSpikesToNodes(int n_spikes, int *node_id)
{
  //std::cout << "n_spikes: " << n_spikes << "\n";
  //for (int i=0; i<n_spikes; i++) {
  //  std::cout << node_id[i] << " ";
  //}
  //std::cout << "\n";

  int *d_node_id;
  gpuErrchk(cudaMalloc(&d_node_id, n_spikes*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_node_id, node_id, n_spikes*sizeof(int),
		       cudaMemcpyHostToDevice));  
  PushSpikeFromRemote<<<(n_spikes+1023)/1024, 1024>>>(n_spikes, d_node_id);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  gpuErrchk(cudaFree(d_node_id));

  return 0;
}

int NeuronGPU::GetExtNeuronInputSpikes(int *n_spikes, int **node, int **port,
				       float **spike_height, bool include_zeros)
{
  ext_neuron_input_spike_node_.clear();
  ext_neuron_input_spike_port_.clear();
  ext_neuron_input_spike_height_.clear();
  
  for (unsigned int i=0; i<node_vect_.size(); i++) {
    if (node_vect_[i]->IsExtNeuron()) {
      int n_node;
      int n_port;
      float *sh = node_vect_[i]->GetExtNeuronInputSpikes(&n_node, &n_port);
      for (int i_neur=0; i_neur<n_node; i_neur++) {
	int i_node = i_neur + node_vect_[i]->i_node_0_;
	for (int i_port=0; i_port<n_port; i_port++) {
	  int j = i_neur*n_port + i_port;
	  if (sh[j] != 0.0 || include_zeros) {
	    ext_neuron_input_spike_node_.push_back(i_node);
	    ext_neuron_input_spike_port_.push_back(i_port);
	    ext_neuron_input_spike_height_.push_back(sh[j]);
	  }
	}
      }	
    }
  }
  *n_spikes = ext_neuron_input_spike_node_.size();
  *node = ext_neuron_input_spike_node_.data();
  *port = ext_neuron_input_spike_port_.data();
  *spike_height = ext_neuron_input_spike_height_.data();
  
  return 0;
}

int NeuronGPU::SetNeuronGroupParam(int i_node, int n_node,
				   std::string param_name, float val)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, n_node, i_group);
  if (i_node_0!=i_node || node_vect_[i_group]->n_node_!=n_node) {
    throw ngpu_exception(std::string("Group parameter ") + param_name
			 + " can only be set for all and only "
			 " the nodes of the same group");
  }
  return node_vect_[i_group]->SetGroupParam(param_name, val);
}

int NeuronGPU::IsNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->IsGroupParam(param_name);
}

float NeuronGPU::GetNeuronGroupParam(int i_node, std::string param_name)
{
  int i_group;
  int i_node_0 = GetNodeSequenceOffset(i_node, 1, i_group);

  return node_vect_[i_group]->GetGroupParam(param_name);
}

std::vector<std::string> NeuronGPU::GetGroupParamNames(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading group parameter names");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetGroupParamNames();
}

int NeuronGPU::GetNGroupParam(int i_node)
{
  if (i_node<0 || i_node>(int)node_group_map_.size()) {
    throw ngpu_exception("Unrecognized node in reading number of "
			 "group parameters");
  }
  int i_group = node_group_map_[i_node];
  
  return node_vect_[i_group]->GetNGroupParam();
}

