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
  spike_generator_ = new SpikeGenerator;
  multimeter_ = new Multimeter;
  net_connection_ = new NetConnection;
  connect_mpi_ = new ConnectMpi;

  SetRandomSeed(54321ULL);

  calibrate_flag_ = false;
  mpi_flag_ = false;
  start_real_time_ = getRealTime();
  max_spike_buffer_size_ = 100;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_neurons_ = 0;
  n_poiss_nodes_ = 0;
  n_spike_gen_nodes_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  /////ConnectMpiInit(&argc, &argv, time_resolution_);
  connect_mpi_->net_connection_ = net_connection_;
  NestedLoop::Init();
}

NeuralGPU::~NeuralGPU()
{
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  delete poiss_generator_;
  delete spike_generator_;
  delete multimeter_;
  for (unsigned int i=0; i<neuron_vect_.size(); i++) {
    delete neuron_vect_[i];
  }
  delete net_connection_;
  delete connect_mpi_;
  FreeNeuronGroupMap();
  FreeGetSpikeArrays();
}

int NeuralGPU::SetRandomSeed(unsigned long long seed)
{
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

int NeuralGPU::CreateNeuron(int n_neurons, int n_receptors)
{
   n_neurons_ = n_neurons;               

  int i_node_0 = net_connection_->connection_.size();
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_neurons, conn);

  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_neurons, conn_node);

  int i_neuron_group = InsertNeuronGroup(n_neurons, n_receptors);
  int n = neuron_vect_.size();
  neuron_vect_[n-1]->Init(i_node_0, n_neurons, n_receptors, i_neuron_group);
  
  return i_node_0;
}

NodeSeq NeuralGPU::CreatePoissonGenerator(int n_nodes, float rate)
{
  CheckUncalibrated("Poisson generator cannot be created after calibration");
  if (n_poiss_nodes_ != 0) {
    std::cerr << "Number of poisson generators cannot be modified.\n";
    exit(0);
  }
  else if (n_nodes <= 0) {
    std::cerr << "Number of nodes must be greater than zero.\n";
    exit(0);
  }

  n_poiss_nodes_ = n_nodes;               

  int i_node_0 = net_connection_->connection_.size();
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >::iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_poiss_nodes_, conn);

  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    std::cerr << "Error: net_connection_.connection_ and "
      "connect_mpi_.extern_connection_ must have the same size!\n";
  }
  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_poiss_nodes_, conn_node);

  float lambda = rate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_poiss_nodes_, lambda);
  int i_neuron_group = InsertNeuronGroup(n_nodes, 0);
  BaseNeuron *bn = new BaseNeuron;
  bn->Init(i_node_0, n_nodes, 0, i_neuron_group);
  neuron_vect_.push_back(bn);
    
  return NodeSeq(i_node_0, n_nodes);
}

NodeSeq NeuralGPU::CreateSpikeGenerator(int n_nodes)
{
  CheckUncalibrated("Spike generator cannot be created after calibration");
  if (n_spike_gen_nodes_ != 0) {
    std::cerr << "Number of spike generators cannot be modified.\n";
    exit(0);
  }
  else if (n_nodes <= 0) {
    std::cerr << "Number of nodes must be greater than zero.\n";
    exit(0);
  }

  n_spike_gen_nodes_ = n_nodes;               

  int i_node_0 = net_connection_->connection_.size();
  
  std::vector<ConnGroup> conn;
  std::vector<std::vector<ConnGroup> >:: iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_spike_gen_nodes_, conn);

  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    std::cerr << "Error: net_connection_.connection_ and "
      "connect_mpi_.extern_connection_ must have the same size!\n";
  }
  std::vector<ExternalConnectionNode > conn_node;
  std::vector<std::vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_spike_gen_nodes_, conn_node);

  spike_generator_->Create(i_node_0, n_spike_gen_nodes_,
			  t_min_, time_resolution_);
  int i_neuron_group = InsertNeuronGroup(n_nodes, 0);
  BaseNeuron *bn = new BaseNeuron;
  bn->Init(i_node_0, n_nodes, 0, i_neuron_group);
  neuron_vect_.push_back(bn);
  
  return NodeSeq(i_node_0, n_nodes);
}

int NeuralGPU::CheckUncalibrated(std::string message)
{
  if (calibrate_flag_ == true) {
    std::cerr << message << "\n";
    exit(0);
  }
  
  return 0;
}

int NeuralGPU::Calibrate()
{
  CheckUncalibrated("Calibration can be made only once");
  calibrate_flag_ = true;
  if (mpi_flag_) {
    std::cout << "Calibrating on host " << connect_mpi_->mpi_id_ << " ...\n";
  }
  else {
    std::cout << "Calibrating ...\n";
  }
  neural_time_ = t_min_;
  
  gpuErrchk(cudaMemcpyToSymbol(NeuralGPUMpiFlag, &mpi_flag_, sizeof(bool)));
	    
  NeuronGroupArrayInit();
  
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
  
  for (unsigned int i=0; i<neuron_vect_.size(); i++) {
    neuron_vect_[i]->Calibrate(t_min_);
  }
  //float x;
  //float y;
  //neuron_vect_[0].GetX(test_arr_idx, 1, &x);
  //neuron_vect_[0].GetY(test_var_idx, test_arr_idx, 1, &y);
  //fprintf(fp,"%f\t%f\n", x, y);

///////////////////////////////////

  return 0;
}

int NeuralGPU::Simulate()
{
  double SpikeBufferUpdate_time = 0;
  double poisson_generator_time = 0;
  double spike_generator_time = 0;
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
    if (n_spike_gen_nodes_>0) {
      spike_generator_->Update(it);
      spike_generator_time += (getRealTime() - time_mark);
    }

    time_mark = getRealTime();
    neural_time_ = neur_t0 + time_resolution_*(it+1);
    for (unsigned int i=0; i<neuron_vect_.size(); i++) {
      neuron_vect_[i]->Update(it, neural_time_);
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
    if (n_spikes > 0) {
      ClearGetSpikeArrays();      
      time_mark = getRealTime();
      NestedLoop::Run(n_spikes, d_SpikeTargetNum);
      NestedLoop_time += (getRealTime() - time_mark);
      time_mark = getRealTime();
      // improve using a grid
      for (unsigned int i=0; i<neuron_vect_.size(); i++) {
	if (neuron_vect_[i]->n_receptors_>0) {
	  GetSpikes<<<(neuron_vect_[i]->n_neurons_
		       *neuron_vect_[i]->n_receptors_+1023)/1024, 1024>>>
	    (neuron_vect_[i]->i_neuron_group_, neuron_vect_[i]->n_neurons_,
	     neuron_vect_[i]->n_receptors_,
	     neuron_vect_[i]->n_var_,
	     neuron_vect_[i]->receptor_weight_arr_,
	     neuron_vect_[i]->receptor_weight_arr_step_,
	     neuron_vect_[i]->receptor_weight_port_step_,
	     neuron_vect_[i]->receptor_input_arr_,
	     neuron_vect_[i]->receptor_input_arr_step_,
	     neuron_vect_[i]->receptor_input_port_step_);
	  
	  gpuErrchk( cudaPeekAtLastError() );
	  gpuErrchk( cudaDeviceSynchronize() );
	}
      }
      GetSpike_time += (getRealTime() - time_mark);
    }
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
  std::cout << "  spike_generator_time: " << spike_generator_time << "\n";
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
			    int *i_neuron_arr, int *i_receptor_arr,
			    int n_neurons)
{
  std::vector<BaseNeuron*> neur_vect;
  std::vector<int> i_neur_vect;
  std::vector<int> i_receptor_vect;
  std::vector<std::string> var_name_vect;
  for (int i=0; i<n_neurons; i++) {
    var_name_vect.push_back(var_name_arr[i]);
    int i_group = neuron_group_map_[i_neuron_arr[i]];
    i_neur_vect.push_back(i_neuron_arr[i] - neuron_vect_[i_group]->i_node_0_);
    i_receptor_vect.push_back(i_receptor_arr[i]);
    neur_vect.push_back(neuron_vect_[i_group]);
  }

  return multimeter_->CreateRecord(neur_vect, file_name, var_name_vect,
  				   i_neur_vect, i_receptor_vect);

}

int NeuralGPU::CreateRecord(std::string file_name, std::string *var_name_arr,
			    int *i_neuron_arr, int n_neurons)
{
  std::vector<int> i_receptor_vect(n_neurons, 0);
  return CreateRecord(file_name, var_name_arr, i_neuron_arr,
		      i_receptor_vect.data(), n_neurons);
}

std::vector<std::vector<float>> *NeuralGPU::GetRecordData(int i_record)
{
  return multimeter_->GetRecordData(i_record);
}

int NeuralGPU::ConnectFixedIndegree
(
 int i_source_neuron_0, int n_source_neurons,
 int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay, int indegree
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  unsigned int *rnd = RandomInt(n_target_neurons*indegree);
  std::vector<int> input_array;
  for (int i=0; i<n_source_neurons; i++) {
    input_array.push_back(i_source_neuron_0 + i);
  }
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source_neurons];
  for (int i=0; i<n_source_neurons; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
  for (int k=0; k<n_target_neurons; k++) {
    for (int i=0; i<indegree; i++) {
      int j = i + rnd[k*indegree+i] % (n_source_neurons - i);
#ifdef _OPENMP
      omp_set_lock(&(lock[i]));
#endif
      if (j!=i) {
#ifdef _OPENMP
	omp_set_lock(&(lock[j]));
#endif
	std::swap(input_array[i], input_array[j]);
#ifdef _OPENMP
	omp_unset_lock(&(lock[j]));
#endif
      }
      int itn = k + i_target_neuron_0;
      int isn = input_array[i];
      net_connection_->Connect(isn, itn, i_port, weight, delay);
#ifdef _OPENMP
      omp_unset_lock(&(lock[i]));
#endif
    }
  }
  delete[] rnd;
#ifdef _OPENMP
  delete[] lock;
#endif
  
  return 0;
}

int NeuralGPU::ConnectAllToAll
(
 int i_source_neuron_0, int n_source_neurons,
 int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source_neurons];
  for (int i=0; i<n_source_neurons; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
  for (int itn=i_target_neuron_0; itn<i_target_neuron_0+n_target_neurons;
       itn++) {
    for (int i=0; i<n_source_neurons; i++) {
      int isn = i_source_neuron_0 + i;
#ifdef _OPENMP
      omp_set_lock(&(lock[i]));
#endif
      net_connection_->Connect(isn, itn, i_port, weight, delay);
#ifdef _OPENMP
      omp_unset_lock(&(lock[i]));
#endif
    }
  }

#ifdef _OPENMP
  delete[] lock;
#endif

  return 0;
}

int NeuralGPU::Connect(int i_source_neuron, int i_target_neuron,
		       unsigned char i_port, float weight, float delay)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  net_connection_->Connect(i_source_neuron, i_target_neuron,
			   i_port, weight, delay);

  return 0;
}

int NeuralGPU::ConnectOneToOne(int i_source_neuron_0, int i_target_neuron_0,
			       int n_neurons, unsigned char i_port,
			       float weight, float delay)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  for (int in=0; in<n_neurons; in++) {
    net_connection_->Connect(i_source_neuron_0+in,i_target_neuron_0+in ,
			    i_port, weight, delay);
  }

  return 0;
}

int NeuralGPU::GetNodeSequenceOffset(int i_node, int n_neurons, int &i_group)
{
  if (i_node<0 || (i_node+n_neurons > (int)neuron_group_map_.size())) {
    std::cerr << "Unrecognized node in setting neuron parameter\n";
    exit(0);
  }
  i_group = neuron_group_map_[i_node];  
  if (neuron_group_map_[i_node+n_neurons-1] != i_group) {
    std::cerr << "Nodes belong to different neuron groups "
      "in setting parameter\n";
    exit(0);
  }
  return neuron_vect_[i_group]->i_node_0_;
}
  
std::vector<int> NeuralGPU::GetNodeArrayWithOffset(int *i_node, int n_neurons,
						   int &i_group)
{
  int in0 = i_node[0];
  if (in0<0 || in0>(int)neuron_group_map_.size()) {
    std::cerr << "Unrecognized node in setting parameter\n";
    exit(0);
  }
  i_group = neuron_group_map_[in0];
  int i0 = neuron_vect_[i_group]->i_node_0_;
  std::vector<int> node_vect;
  node_vect.assign(i_node, i_node+n_neurons);
  for(int i=0; i<n_neurons; i++) {
    int in = node_vect[i];
    if (in<0 || in>=(int)neuron_group_map_.size()) {
      std::cerr << "Unrecognized node in setting parameter\n";
      exit(0);
    }
    if (neuron_group_map_[in] != i_group) {
      std::cerr << "Nodes belong to different neuron groups "
	"in setting parameter\n";
      exit(0);
    }
    node_vect[i] -= i0;
  }
  return node_vect;
}

int NeuralGPU::SetNeuronParam(std::string param_name, int i_node,
			       int n_neurons, float val)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_neurons, i_group);
  
  return neuron_vect_[i_group]->SetScalParam(param_name, i_neuron,
					      n_neurons, val);
}

int NeuralGPU::SetNeuronParam(std::string param_name, int *i_node,
			       int n_neurons, float val)
{
  int i_group;
  std::vector<int> node_vect = GetNodeArrayWithOffset(i_node, n_neurons,
						      i_group);
  return neuron_vect_[i_group]->SetScalParam(param_name, node_vect.data(),
					      n_neurons, val);
}

int NeuralGPU::SetNeuronParam(std::string param_name, int i_node,
			       int n_neurons, float *params, int vect_size)
{
  int i_group;
  int i_neuron = i_node - GetNodeSequenceOffset(i_node, n_neurons, i_group);
  
  return neuron_vect_[i_group]->SetVectParam(param_name, i_neuron,
					     n_neurons, params, vect_size);
}

int NeuralGPU::SetNeuronParam(std::string param_name, int *i_node,
			       int n_neurons, float *params, int vect_size)
{
  int i_group;
  std::vector<int> node_vect = GetNodeArrayWithOffset(i_node, n_neurons,
						      i_group);
  
  return neuron_vect_[i_group]->SetVectParam(param_name, node_vect.data(),
					      n_neurons, params, vect_size);
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

int NeuralGPU::SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
				 float *spike_height)
{
  return spike_generator_->Set(i_node, n_spikes, spike_time, spike_height);
}

int NeuralGPU::RemoteConnectFixedIndegree
(int i_source_host, int i_source_neuron_0, int n_source_neurons,
 int i_target_host, int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay, int indegree
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  if (MpiId()==i_source_host && i_source_host==i_target_host) {
    return ConnectFixedIndegree(i_source_neuron_0, n_source_neurons, i_target_neuron_0,
			 n_target_neurons, i_port, weight, delay, indegree);
  }
  else if (MpiId()==i_source_host || MpiId()==i_target_host) {
    int *i_remote_neuron_arr = new int[n_target_neurons*indegree];
    int i_new_remote_neuron;
    if (MpiId() == i_target_host) {
      i_new_remote_neuron = net_connection_->connection_.size();
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_source_host);
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_source_host);
      std::vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
      
      //NEW, CHECK ///////////
      InsertNeuronGroup(i_new_remote_neuron
			- net_connection_->connection_.size(), 0);
      ///////////////////////
      
      connect_mpi_->MPI_Recv_int(i_remote_neuron_arr, n_target_neurons*indegree, i_source_host);

      for (int k=0; k<n_target_neurons; k++) {
	for (int i=0; i<indegree; i++) {
      	  int i_remote_neuron = i_remote_neuron_arr[k*indegree+i];
	  int i_target_neuron = k + i_target_neuron_0;
	  net_connection_->Connect(i_remote_neuron, i_target_neuron, i_port, weight, delay);
	}
      }
    }
    else if (MpiId() == i_source_host) {
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_target_host);
      unsigned int *rnd = RandomInt(n_target_neurons*indegree); // check parall. seed problem
      std::vector<int> input_array;
      for (int i=0; i<n_source_neurons; i++) {
	input_array.push_back(i_source_neuron_0 + i);
      }
      for (int k=0; k<n_target_neurons; k++) {
	for (int i=0; i<indegree; i++) {
	  int j = i + rnd[k*indegree+i] % (n_source_neurons - i);
	  if (j!=i) {
	    std::swap(input_array[i], input_array[j]);
	  }
	  int i_source_neuron = input_array[i];
	  
	  int i_remote_neuron = -1;
	  for (std::vector<ExternalConnectionNode >::iterator it =
		 connect_mpi_->extern_connection_[i_source_neuron].begin();
	       it <  connect_mpi_->extern_connection_[i_source_neuron].end(); it++) {
	    if ((*it).target_host_id == i_target_host) {
	      i_remote_neuron = (*it).remote_neuron_id;
	      break;
	    }
	  }
	  if (i_remote_neuron == -1) {
	    i_remote_neuron = i_new_remote_neuron;
	    i_new_remote_neuron++;
	    ExternalConnectionNode conn_node = {i_target_host, i_remote_neuron};
	    connect_mpi_->extern_connection_[i_source_neuron].push_back(conn_node);
	  }
	  i_remote_neuron_arr[k*indegree+i] = i_remote_neuron;
	}
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_target_host);
      connect_mpi_->MPI_Send_int(i_remote_neuron_arr, n_target_neurons*indegree, i_target_host);
      delete[] rnd;
    }
    delete[] i_remote_neuron_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

int NeuralGPU::RemoteConnectAllToAll
(
 int i_source_host, int i_source_neuron_0, int n_source_neurons,
 int i_target_host, int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  if (MpiId()==i_source_host && i_source_host==i_target_host) {
    return ConnectAllToAll(i_source_neuron_0, n_source_neurons, i_target_neuron_0,
			 n_target_neurons, i_port, weight, delay);
  }
  else if (MpiId()==i_source_host || MpiId()==i_target_host) {
    int *i_remote_neuron_arr = new int[n_target_neurons*n_source_neurons];
    int i_new_remote_neuron;
    if (MpiId() == i_target_host) {
      i_new_remote_neuron = net_connection_->connection_.size();
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_source_host);
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_source_host);
      std::vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
            
      //NEW, CHECK ///////////
      InsertNeuronGroup(i_new_remote_neuron
			- net_connection_->connection_.size(), 0);
      ///////////////////////
      
      connect_mpi_->MPI_Recv_int(i_remote_neuron_arr, n_target_neurons*n_source_neurons,
				 i_source_host);

      for (int k=0; k<n_target_neurons; k++) {
	for (int i=0; i<n_source_neurons; i++) {
      	  int i_remote_neuron = i_remote_neuron_arr[k*n_source_neurons+i];
	  int i_target_neuron = k + i_target_neuron_0;
	  net_connection_->Connect(i_remote_neuron, i_target_neuron, i_port, weight, delay);
	}
      }
    }
    else if (MpiId() == i_source_host) {
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_target_host);
      for (int k=0; k<n_target_neurons; k++) {
	for (int i=0; i<n_source_neurons; i++) {
	  int i_source_neuron = i + i_source_neuron_0;
	  
	  int i_remote_neuron = -1;
	  for (std::vector<ExternalConnectionNode >::iterator it =
		 connect_mpi_->extern_connection_[i_source_neuron].begin();
	       it <  connect_mpi_->extern_connection_[i_source_neuron].end(); it++) {
	    if ((*it).target_host_id == i_target_host) {
	      i_remote_neuron = (*it).remote_neuron_id;
	      break;
	    }
	  }
	  if (i_remote_neuron == -1) {
	    i_remote_neuron = i_new_remote_neuron;
	    i_new_remote_neuron++;
	    ExternalConnectionNode conn_node = {i_target_host, i_remote_neuron};
	    connect_mpi_->extern_connection_[i_source_neuron].push_back(conn_node);
	  }
	  i_remote_neuron_arr[k*n_source_neurons+i] = i_remote_neuron;
	}
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_target_host);
      connect_mpi_->MPI_Send_int(i_remote_neuron_arr, n_target_neurons*n_source_neurons,
				 i_target_host);
    }
    delete[] i_remote_neuron_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

int NeuralGPU::RemoteConnectOneToOne
(
 int i_source_host, int i_source_neuron_0,
 int i_target_host, int i_target_neuron_0, int n_neurons,
 unsigned char i_port, float weight, float delay
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  if (MpiId()==i_source_host && i_source_host==i_target_host) {
    return ConnectOneToOne(i_source_neuron_0, i_target_neuron_0,
			 n_neurons, i_port, weight, delay);
  }
  else if (MpiId()==i_source_host || MpiId()==i_target_host) {
    int *i_remote_neuron_arr = new int[n_neurons];
    int i_new_remote_neuron;
    if (MpiId() == i_target_host) {
      i_new_remote_neuron = net_connection_->connection_.size();
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_source_host);
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_source_host);
      std::vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
            
      //NEW, CHECK ///////////
      InsertNeuronGroup(i_new_remote_neuron
			- net_connection_->connection_.size(), 0);
      ///////////////////////
      
      connect_mpi_->MPI_Recv_int(i_remote_neuron_arr, n_neurons, i_source_host);

      for (int i=0; i<n_neurons; i++) {
	int i_remote_neuron = i_remote_neuron_arr[i];
	int i_target_neuron = i + i_target_neuron_0;
	net_connection_->Connect(i_remote_neuron, i_target_neuron, i_port, weight, delay);
      }
    }
    else if (MpiId() == i_source_host) {
      connect_mpi_->MPI_Recv_int(&i_new_remote_neuron, 1, i_target_host);
      for (int i=0; i<n_neurons; i++) {
	int i_source_neuron = i + i_source_neuron_0;
	  
	int i_remote_neuron = -1;
	for (std::vector<ExternalConnectionNode >::iterator it =
	       connect_mpi_->extern_connection_[i_source_neuron].begin();
	     it <  connect_mpi_->extern_connection_[i_source_neuron].end(); it++) {
	  if ((*it).target_host_id == i_target_host) {
	    i_remote_neuron = (*it).remote_neuron_id;
	    break;
	  }
	}
	if (i_remote_neuron == -1) {
	  i_remote_neuron = i_new_remote_neuron;
	  i_new_remote_neuron++;
	  ExternalConnectionNode conn_node = {i_target_host, i_remote_neuron};
	  connect_mpi_->extern_connection_[i_source_neuron].push_back(conn_node);
	}
	i_remote_neuron_arr[i] = i_remote_neuron;
      }
      connect_mpi_->MPI_Send_int(&i_new_remote_neuron, 1, i_target_host);
      connect_mpi_->MPI_Send_int(i_remote_neuron_arr, n_neurons, i_target_host);
    }
    delete[] i_remote_neuron_arr;
  }
  MPI_Barrier( MPI_COMM_WORLD );

  return 0;
}

int NeuralGPU::RemoteConnect(int i_source_host, int i_source_neuron,
			     int i_target_host, int i_target_neuron,
			     unsigned char i_port, float weight, float delay)
{
  CheckUncalibrated("Connections cannot be created after calibration");
  return connect_mpi_->RemoteConnect(i_source_host, i_source_neuron,
				     i_target_host, i_target_neuron,
				     i_port, weight, delay);
}

int NeuralGPU::ConnectFixedIndegreeArray
(
 int i_source_neuron_0, int n_source_neurons,
 int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float *weight_arr, float *delay_arr, int indegree
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  unsigned int *rnd = RandomInt(n_target_neurons*indegree);
  std::vector<int> input_array;
  for (int i=0; i<n_source_neurons; i++) {
    input_array.push_back(i_source_neuron_0 + i);
  }
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source_neurons];
  for (int i=0; i<n_source_neurons; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared) collapse(2)
#endif
  for (int k=0; k<n_target_neurons; k++) {
    for (int i=0; i<indegree; i++) {
      int j = i + rnd[k*indegree+i] % (n_source_neurons - i);
#ifdef _OPENMP
      omp_set_lock(&(lock[i]));
#endif
      if (j!=i) {
#ifdef _OPENMP
	omp_set_lock(&(lock[j]));
#endif
	std::swap(input_array[i], input_array[j]);
#ifdef _OPENMP
	omp_unset_lock(&(lock[j]));
#endif
      }
      int itn = k + i_target_neuron_0;
      int isn = input_array[i];
      size_t i_arr = (size_t)k*indegree + i;
      net_connection_->Connect(isn, itn, i_port, weight_arr[i_arr],
			       delay_arr[i_arr]);
#ifdef _OPENMP
      omp_unset_lock(&(lock[i]));
#endif
    }
  }
  delete[] rnd;
#ifdef _OPENMP
  delete[] lock;
#endif
  
  return 0;
}

int NeuralGPU::ConnectFixedTotalNumberArray
(
 int i_source_neuron_0, int n_source_neurons,
 int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float *weight_arr, float *delay_arr, int n_conn
 )
{
  CheckUncalibrated("Connections cannot be created after calibration");
  unsigned int *rnd = RandomInt(2*n_conn);
#ifdef _OPENMP
  omp_lock_t *lock = new omp_lock_t[n_source_neurons];
  for (int i=0; i<n_source_neurons; i++) {
    omp_init_lock(&(lock[i]));
  }
#pragma omp parallel for default(shared)
#endif
  for (int i_conn=0; i_conn<n_conn; i_conn++) {
    int i = rnd[2*i_conn] % n_source_neurons;
    int j = rnd[2*i_conn+1] % n_target_neurons;
    int isn = i + i_source_neuron_0;
    int itn = j + i_target_neuron_0;
#ifdef _OPENMP
    omp_set_lock(&(lock[i]));
#endif
    net_connection_->Connect(isn, itn, i_port, weight_arr[i_conn],
                             delay_arr[i_conn]);
#ifdef _OPENMP
      omp_unset_lock(&(lock[i]));
#endif
  }
  delete[] rnd;
#ifdef _OPENMP
  delete[] lock;
#endif
  
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

