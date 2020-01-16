import sys
import ctypes
import neuralgpu as ngpu
from random import randrange


ngpu.ConnectMpiInit();
mpi_np = ngpu.MpiNp()

if (mpi_np != 2) | (len(sys.argv) != 2):
    print ("Usage: mpirun -np 2 python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])/5

mpi_id = ngpu.MpiId()
print("Building on host ", mpi_id, " ...")
  
ngpu.SetRandomSeed(1234) # seed for GPU random numbers

n_receptors = 2

delay = 1.0       # synaptic delay in ms

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

CPN = 1000 # number of output connections per neuron

fext = 0.25 # fraction of the excitatory neurons that
# send their output to neurons of another mpi host
NEext = (int)(fext*NE)
NEint = NE - NEext

Wex = 0.05
Win = 0.35

# poisson generator parameters
poiss_rate = 20000.0 # poisson signal rate in Hz
poiss_weight = 0.37
poiss_delay = 0.2 # poisson signal delay in ms
n_pg = n_neurons  # number of poisson generators
# create poisson generator
pg = ngpu.CreatePoissonGenerator(n_pg, poiss_rate)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("aeif_cond_beta", n_neurons, n_receptors)
excint_neuron = neuron[0:NEint-1]      # excitatory group
# of neurons that project internally
excest_neuron = neuron[NEint:NE-1]      # excitatory group
# of neurons that project externally

inh_neuron = neuron[NE:n_neurons-1]   # inhibitory neuron group
  
# receptor parameters
E_rev = [0.0, -85.0]
taus_decay = [1.0, 1.0]
taus_rise = [1.0, 1.0]
ngpu.SetStatus(neuron, {"E_rev":E_rev, "taus_decay":taus_decay,
                        "taus_rise":taus_rise})

# Excitatory local connections, defined on all hosts
# connect excitatory neurons to port 0 of all neurons
# weight Wex and fixed outdegree CPN

exc_conn_dict={"rule": "fixed_outdegree", "outdegree": CPN}
exc_syn_dict={"weight": Wex, "delay": delay, "receptor":0}
ngpu.Connect(excint_neuron, neuron, exc_conn_dict, exc_syn_dict)


# Inhibitory local connections, defined on all hosts
# connect inhibitory neurons to port 1 of all neurons
# weight Win and fixed outdegree CPN

inh_conn_dict={"rule": "fixed_outdegree", "outdegree": CPN}
inh_syn_dict={"weight": Win, "delay": delay, "receptor":1}
ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)


#connect poisson generator to port 0 of all neurons
pg_conn_dict={"rule": "one_to_one"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay, "receptor":0}
ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

filename = "test_brunel_outdegree_mpi" + str(mpi_id) + ".dat"

# any set of neuron indexes
i_neuron_arr = [neuron[0], neuron[randrange(n_neurons)],
                neuron[randrange(n_neurons)], neuron[randrange(n_neurons)],
                neuron[n_neurons-1]]
i_receptor_arr = [0, 0, 0, 0, 0]

# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m", "V_m", "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                i_receptor_arr)

######################################################################
## WRITE HERE REMOTE CONNECTIONS
######################################################################

# Excitatory remote connections
# connect excitatory neurons to port 0 of all neurons
# weight Wex and fixed indegree CPN
# host 0 to host 1
ngpu.RemoteConnect(0, excest_neuron, 1, neuron, exc_conn_dict, exc_syn_dict)
# host 1 to host 0
ngpu.RemoteConnect(1, excest_neuron, 0, neuron, exc_conn_dict, exc_syn_dict)

ngpu.Simulate()

nrows=ngpu.GetRecordDataRows(record)
ncol=ngpu.GetRecordDataColumns(record)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]
V4=[row[4] for row in data_list]
V5=[row[5] for row in data_list]

import matplotlib.pyplot as plt

fig1 = plt.figure(1)
fig1.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V1)

fig2 = plt.figure(2)
fig2.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V2)

fig3 = plt.figure(3)
fig3.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V3)

fig4 = plt.figure(4)
fig4.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V4)

fig5 = plt.figure(5)
fig5.suptitle("host " + str(mpi_id), fontsize=20)
plt.plot(t, V5)

plt.show()
