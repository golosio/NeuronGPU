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

#include <iostream>
#include <string>
#include <algorithm>
#include <mpi.h>
#include <curand.h>
//#include "connect.h"
#include "spike_buffer.h"
#include "rk5.h"
#include "cuda_error.h"
#include "aeif.h"
#include "send_spike.h"
#include "get_spike.h"
#include "connect_mpi.h"
#include "spike_mpi.h"
#include "spike_generator.h"
#include "multimeter.h"
#include "prefix_scan.h"
#include "getRealTime.h"
#include "random.h"
#include "neural_gpu.h"

#ifdef _OPENMP
#include <omp.h>
#define THREAD_MAXNUM omp_get_max_threads()
#define THREAD_IDX omp_get_thread_num()
#else
#define THREAD_MAXNUM 1
#define THREAD_IDX 0
#endif

using namespace std;

#define VERBOSE_TIME

NeuralGPU::NeuralGPU()
{
  random_generator_ = new curandGenerator_t;
  CURAND_CALL(curandCreateGenerator(random_generator_,
				    CURAND_RNG_PSEUDO_DEFAULT));
  poiss_generator_ = new PoissonGenerator;
  spike_generator_ = new SpikeGenerator;
  multimeter_ = new Multimeter;
  aeif_ = new AEIF;
  net_connection_ = new NetConnection;
  connect_mpi_ = new ConnectMpi;
  prefix_scan_ = new PrefixScan;

  SetRandomSeed(54321ULL);
  
  start_real_time_ = getRealTime();
  max_spike_buffer_num_ = 100;
  t_min_ = 0.0;
  sim_time_ = 1000.0;        //Simulation time in ms
  n_neurons_ = 0;
  n_poiss_nodes_ = 0;
  n_spike_gen_nodes_ = 0;
  SetTimeResolution(0.1);  // time resolution in ms
  /////ConnectMpiInit(&argc, &argv, time_resolution_);
  connect_mpi_->net_connection_ = net_connection_;
  prefix_scan_->Init();
}

NeuralGPU::~NeuralGPU()
{
  CURAND_CALL(curandDestroyGenerator(*random_generator_));
  delete poiss_generator_;
  delete spike_generator_;
  delete multimeter_;
  delete aeif_;
  delete net_connection_;
  delete connect_mpi_;
  delete prefix_scan_;
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

int NeuralGPU::CreateNeuron(int n_neurons, int n_receptors)
{
  if (n_neurons_ != 0) {
    cerr << "Number of neurons cannot be modified.\n";
    exit(0);
  }
  else if (n_neurons <= 0) {
    cerr << "Number of neurons must be greater than zero.\n";
    exit(0);
  }
  else if (n_receptors <= 0) {
    cerr << "Number of receptors must be greater than zero.\n";
    exit(0);
  }

  n_neurons_ = n_neurons;               

  int i_node_0 = net_connection_->connection_.size();
  
  vector<ConnGroup> conn;
  vector<vector<ConnGroup> >:: iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_neurons_, conn);

  vector<ExternalConnectionNode > conn_node;
  vector<vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_neurons_, conn_node);

  //SpikeInit(max_spike_num_);
  aeif_->Init(i_node_0, n_neurons_, n_receptors);
  
  return i_node_0;
}

int NeuralGPU::CreatePoissonGenerator(int n_nodes, float rate)
{
  if (n_poiss_nodes_ != 0) {
    cerr << "Number of poisson generators cannot be modified.\n";
    exit(0);
  }
  else if (n_nodes <= 0) {
    cerr << "Number of nodes must be greater than zero.\n";
    exit(0);
  }

  n_poiss_nodes_ = n_nodes;               

  int i_node_0 = net_connection_->connection_.size();
  
  vector<ConnGroup> conn;
  vector<vector<ConnGroup> >:: iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_poiss_nodes_, conn);

  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    cerr << "Error: net_connection_.connection_ and "
      "connect_mpi_.extern_connection_ must have the same size!\n";
  }
  vector<ExternalConnectionNode > conn_node;
  vector<vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_poiss_nodes_, conn_node);

  float lambda = rate*time_resolution_ / 1000.0; // rate is in Hz, time in ms
  poiss_generator_->Create(random_generator_, i_node_0, n_poiss_nodes_, lambda);
  
  return i_node_0;
}

int NeuralGPU::CreateSpikeGenerator(int n_nodes)
{
  if (n_spike_gen_nodes_ != 0) {
    cerr << "Number of spike generators cannot be modified.\n";
    exit(0);
  }
  else if (n_nodes <= 0) {
    cerr << "Number of nodes must be greater than zero.\n";
    exit(0);
  }

  n_spike_gen_nodes_ = n_nodes;               

  int i_node_0 = net_connection_->connection_.size();
  
  vector<ConnGroup> conn;
  vector<vector<ConnGroup> >:: iterator it
    = net_connection_->connection_.end();
  net_connection_->connection_.insert(it, n_spike_gen_nodes_, conn);

  if ((int)connect_mpi_->extern_connection_.size() != i_node_0) {
    cerr << "Error: net_connection_.connection_ and "
      "connect_mpi_.extern_connection_ must have the same size!\n";
  }
  vector<ExternalConnectionNode > conn_node;
  vector<vector< ExternalConnectionNode> >::iterator it1
    = connect_mpi_->extern_connection_.end();
  connect_mpi_->extern_connection_.insert(it1, n_spike_gen_nodes_, conn_node);

  spike_generator_->Create(i_node_0, n_spike_gen_nodes_,
			  t_min_, time_resolution_);
  
  return i_node_0;
}

int NeuralGPU::Simulate()
{
  double SpikeBufferUpdate_time = 0;
  double poisson_generator_time = 0;
  double spike_generator_time = 0;
  double aeif_Update_time = 0;
  double copy_ext_spike_time = 0;
  double SendExternalSpike_time = 0;
  double SendSpikeToRemote_time = 0;
  double RecvSpikeFromRemote_time = 0;
  double PrefixScan_time = 0;
  double GetSpike_time = 0;
  double SpikeReset_time = 0;
  double ExternalSpikeReset_time = 0;
  double time_mark;
  
  float t_min = 0.0;

  max_spike_num_ = net_connection_->connection_.size();
  max_spike_per_host_ = net_connection_->connection_.size();

  SpikeInit(max_spike_num_);
  SpikeBufferInit(net_connection_, max_spike_buffer_num_);

  // remove superfluous argument mpi_np
  connect_mpi_->ExternalSpikeInit(connect_mpi_->extern_connection_.size(),
				 max_spike_num_, connect_mpi_->mpi_np_,
				 max_spike_per_host_);
  InitGetSpikeArray(n_neurons_, aeif_->n_receptors_);

  //////////////////////////////////////////////////
  //char filename[100];
  //sprintf(filename, "test_arr_%d.dat", connect_mpi_.mpi_id_);
  //FILE *fp=fopen(filename, "wb");
  multimeter_->OpenFiles();
  
  int Nt=(int)round(sim_time_/time_resolution_);
  printf("%d\n", Nt);

  aeif_->Calibrate(t_min);

  //float x;
  //float y;
  //aeif_.GetX(test_arr_idx, 1, &x);
  //aeif_.GetY(test_var_idx, test_arr_idx, 1, &y);
  //fprintf(fp,"%f\t%f\n", x, y);

///////////////////////////////////
  multimeter_->WriteRecords();
  
  build_real_time_ = getRealTime();
  
  cout << "Simulating on host " << connect_mpi_->mpi_id_ << " ..." <<endl;

  for (int it=0; it<Nt; it++) {
    float t1 = t_min_ + time_resolution_*(it + 1);
    if (it%100==0)
      printf("%d\n", it);

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
    aeif_->Update(it, t1);
    aeif_Update_time += (getRealTime() - time_mark);
    
    multimeter_->WriteRecords();
    
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

    int n_spikes;
    time_mark = getRealTime();
    gpuErrchk(cudaMemcpy(&n_spikes, d_SpikeNum, sizeof(int),
			 cudaMemcpyDeviceToHost));
    //cout << "n_spikes: " << n_spikes << endl;
    if (n_spikes > 0) {
      prefix_scan_->Scan(d_SpikeTargetNumSum, d_SpikeTargetNum, n_spikes);
      uint n_get_spikes;
      gpuErrchk(cudaMemcpy(&n_get_spikes, &d_SpikeTargetNumSum[n_spikes],
			   sizeof(uint), cudaMemcpyDeviceToHost));
      PrefixScan_time += (getRealTime() - time_mark);
      if(n_get_spikes>0) {
        ClearGetSpikeArray(n_neurons_, aeif_->n_receptors_);

	time_mark = getRealTime();
	uint grid_dim_x, grid_dim_y;
	if (n_get_spikes<65536*1024) { // max grid dim * max block dim
	  grid_dim_x = (n_get_spikes+1023)/1024;
	  grid_dim_y = 1;
	}
	else {
	  grid_dim_x = 64; // I think it's not necessary to increase it
	  if (n_get_spikes>grid_dim_x*1024*65535) {
	    cerr << "n_get_spikes " << n_get_spikes << " larger than threshold "
		 << grid_dim_x*1024*65535 << " .\n";
	    exit(-1);
	  }
	  grid_dim_y = (n_get_spikes + grid_dim_x*1024 -1) / (grid_dim_x*1024);
	}
	dim3 numBlocks(grid_dim_x, grid_dim_y);
	CollectSpikes<<<numBlocks, 1024>>>(n_get_spikes, aeif_->n_var_,
				      aeif_->n_params_);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

        // improve using a grid
        GetSpikes<<<(n_neurons_*aeif_->n_receptors_+1023)/1024, 1024>>>
                    (aeif_->n_receptors_);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	GetSpike_time += (getRealTime() - time_mark);
      }
    }
    time_mark = getRealTime();
    SpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SpikeReset_time += (getRealTime() - time_mark);
    
    time_mark = getRealTime();
    ExternalSpikeReset<<<1, 1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    ExternalSpikeReset_time += (getRealTime() - time_mark);
  }
  end_real_time_ = getRealTime();
  //fclose(fp);
  multimeter_->CloseFiles();
  //aeif.rk5.Free();

#ifdef VERBOSE_TIME
  cout << endl;
  cout << "  SpikeBufferUpdate_time: " << SpikeBufferUpdate_time << endl;
  cout << "  poisson_generator_time: " << poisson_generator_time << endl;
  cout << "  spike_generator_time: " << spike_generator_time << endl;
  cout << "  aeif_Update_time: " << aeif_Update_time << endl;
  cout << "  copy_ext_spike_time: " << copy_ext_spike_time << endl;
  cout << "  SendExternalSpike_time: " << SendExternalSpike_time << endl;
  cout << "  SendSpikeToRemote_time: " << SendSpikeToRemote_time << endl;
  cout << "  RecvSpikeFromRemote_time: " << RecvSpikeFromRemote_time << endl;
  cout << "  PrefixScan_time: " << PrefixScan_time << endl;
  cout << "  GetSpike_time: " << GetSpike_time << endl;
  cout << "  SpikeReset_time: " << SpikeReset_time << endl;
  cout << "  ExternalSpikeReset_time: " << ExternalSpikeReset_time << endl;
#endif
  printf("Build real time = %lf\n",
	 (build_real_time_ - start_real_time_));
  printf("Simulation real time = %lf\n",
	 (end_real_time_ - build_real_time_));

  return 0;
}

int NeuralGPU::CreateRecord(string file_name, string var_name, int *i_neurons,
			    int n_neurons)
{
  return multimeter_->CreateRecord(aeif_, file_name, var_name, i_neurons,
				  n_neurons);
}

int NeuralGPU::ConnectFixedIndegree
(
 int i_source_neuron_0, int n_source_neurons,
 int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay, int indegree
 )
{
  unsigned int *rnd = RandomInt(n_target_neurons*indegree);
  vector<int> input_array;
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
	swap(input_array[i], input_array[j]);
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

int NeuralGPU::ConnectOneToOne
(
 int i_source_neuron_0, int i_target_neuron_0, int n_neurons,
 unsigned char i_port, float weight, float delay
 )
{
  for (int in=0; in<n_neurons; in++) {
    net_connection_->Connect(i_source_neuron_0+in,i_target_neuron_0+in ,
			    i_port, weight, delay);
  }

  return 0;
}

int NeuralGPU::SetNeuronParams(string param_name, int i_node, int n_neurons,
			       float val)
{
  int i_neuron = i_node - aeif_->i_node_0_;
  
  return aeif_->SetParams(param_name, i_neuron, n_neurons, val);
}

int NeuralGPU::SetNeuronVectParams(string param_name, int i_node, int n_neurons,
				   float *params, int vect_size)
{
  int i_neuron = i_node - aeif_->i_node_0_;
  
  return aeif_->SetVectParams(param_name, i_neuron, n_neurons, params,
			     vect_size);
}

int NeuralGPU::ConnectMpiInit(int argc, char *argv[])
{
  return connect_mpi_->MpiInit(argc, argv);
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
  return MPI_Finalize();
}

int NeuralGPU::SetSpikeGenerator(int i_node, int n_spikes, float *spike_time,
				 float *spike_height)
{
  return spike_generator_->Set(i_node, n_spikes, spike_time, spike_height);
}

unsigned int *NeuralGPU::RandomInt(size_t n)
{
  return curand_int(*random_generator_, n);
}


int NeuralGPU::RemoteConnectFixedIndegree
(
 int i_source_host, int i_source_neuron_0, int n_source_neurons,
 int i_target_host, int i_target_neuron_0, int n_target_neurons,
 unsigned char i_port, float weight, float delay, int indegree
 )
{
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
      vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
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
      vector<int> input_array;
      for (int i=0; i<n_source_neurons; i++) {
	input_array.push_back(i_source_neuron_0 + i);
      }
      for (int k=0; k<n_target_neurons; k++) {
	for (int i=0; i<indegree; i++) {
	  int j = i + rnd[k*indegree+i] % (n_source_neurons - i);
	  if (j!=i) {
	    swap(input_array[i], input_array[j]);
	  }
	  int i_source_neuron = input_array[i];
	  
	  int i_remote_neuron = -1;
	  for (vector<ExternalConnectionNode >::iterator it =
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
      vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
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
	  for (vector<ExternalConnectionNode >::iterator it =
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
      vector<ConnGroup> conn;
      net_connection_->connection_.insert(net_connection_->connection_.end(),
					  i_new_remote_neuron
					  - net_connection_->connection_.size(), conn);
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
	for (vector<ExternalConnectionNode >::iterator it =
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
  return connect_mpi_->RemoteConnect(i_source_host, i_source_neuron,
				     i_target_host, i_target_neuron,
				     i_port, weight, delay);
}

