import sys
import ctypes
import neuralgpu as ngpu
from random import randrange

if len(sys.argv) != 2:
    print ("Usage: python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])/5

print("Building ...")
  
n_receptors = 2

delay = 1.0       # synaptic delay in ms

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

CE = 800   # number of excitatory synapses per neuron
CI = CE/4  # number of inhibitory synapses per neuron

Wex = 0.05
Win = 0.35

# poisson generator parameters
poiss_rate = 20000.0 # poisson signal rate in Hz
poiss_weight = 0.369
poiss_delay = 0.2 # poisson signal delay in ms
n_pg = n_neurons  # number of poisson generators
# create poisson generator
pg = ngpu.CreatePoissonGenerator(n_pg, poiss_rate)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.CreateNeuron("AEIF", n_neurons, n_receptors)
exc_neuron = neuron      # excitatory neuron id
inh_neuron = neuron + NE # inhibitory neuron id
  
# receptor parameters
E_rev = [0.0, -85.0]
taus_decay = [1.0, 1.0]
taus_rise = [1.0, 1.0]

ngpu.SetNeuronVectParams("E_rev", neuron, n_neurons, E_rev)
ngpu.SetNeuronVectParams("taus_decay", neuron, n_neurons, taus_decay)
ngpu.SetNeuronVectParams("taus_rise", neuron, n_neurons, taus_rise)
mean_delay = 0.5
std_delay = 0.25
min_delay = 0.1
# Excitatory connections
# connect excitatory neurons to port 0 of all neurons
# normally distributed delays, weight Wex and CE connections per neuron
exc_delays = ngpu.RandomNormalClipped(CE*n_neurons, mean_delay,
  			              std_delay, min_delay,
  			              mean_delay+3*std_delay)

# efficient way to build float array with equal elements
exc_weights = (ctypes.c_float * (CE*n_neurons))(*([Wex] * (CE*n_neurons)))
ngpu.ConnectFixedIndegreeArray(exc_neuron, NE, neuron, n_neurons,
			       0, exc_weights, exc_delays, CE)

# Inhibitory connections
# connect inhibitory neurons to port 1 of all neurons
# normally distributed delays, weight Win and CI connections per neuron
inh_delays = ngpu.RandomNormalClipped(CI*n_neurons, mean_delay,
  					    std_delay, min_delay,
  					    mean_delay+3*std_delay)

# efficient way to build float array with equal elements
inh_weights = (ctypes.c_float * (CI*n_neurons))(*([Win] * (CI*n_neurons)))
ngpu.ConnectFixedIndegreeArray(inh_neuron, NI, neuron, n_neurons,
				  1, inh_weights, inh_delays, CI)

#connect poisson generator to port 0 of all neurons
ngpu.ConnectOneToOne(pg, neuron, n_neurons, 0, poiss_weight,
			   poiss_delay)

filename = "test_brunel_net.dat"
i_neuron_arr = [neuron, neuron+randrange(n_neurons), neuron+n_neurons-1]
i_receptor_arr = [0, 0, 0]
# any set of neuron indexes
# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                i_receptor_arr)
# just to have same results in different simulations
ngpu.SetRandomSeed(1234)
ngpu.SetMaxSpikeBufferSize(10) # spike buffer per neuron size

ngpu.Simulate()

nrows=ngpu.GetRecordDataRows(record)
ncol=ngpu.GetRecordDataColumns(record)
#print nrows, ncol

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)

plt.figure(2)
plt.plot(t, V2)

plt.figure(3)
plt.plot(t, V3)

plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close()
