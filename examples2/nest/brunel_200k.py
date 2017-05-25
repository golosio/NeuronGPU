
import nest
import time

nest.ResetKernel()

nest.SetKernelStatus({"local_num_threads": 8})

start_time = time.time()

time_resolution = 0.1

sim_time = 1000.0

delay = 1.0 # synaptic delay in ms

order = 40000
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
n_neurons = NE + NI   # number of neurons in total

CE = 800  # number of excitatory synapses per neuron
CI = CE/4  # number of inhibitory synapses per neuron


Wex = 0.04995

Win = -0.35 #-0.65:29.30
#old -0.65:30.0, -0.55: 30.20, -0.45: 30.40, -0.4:30.60, -0.3: 31.60, -0.25: 32.80, -0.2: 34.70, -0.1: 43.90

neuron_params = {
                 "tau_syn_ex": 1.0,
                 "tau_syn_in": 1.0,
                 "V_reset": -60.0
}


nest.SetKernelStatus({"resolution": time_resolution, "print_time": True,
                      "overwrite_files": True})

print("Building ...")

nest.SetDefaults("aeif_cond_alpha", neuron_params)

nodes_ex = nest.Create("aeif_cond_alpha", NE)
nodes_in = nest.Create("aeif_cond_alpha", NI)

nest.CopyModel("static_synapse", "excitatory",
               {"weight": Wex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory",
               {"weight": Win, "delay": delay})


from numpy import loadtxt
from numpy import arange

pg = nest.Create("poisson_generator")
nest.SetStatus(pg, {"rate": 20000.0})

#poiss=loadtxt('poisson.dat')
##poiss=poiss[0:10000]
#sg = nest.Create("spike_generator")
#nest.SetStatus(sg, [{"spike_times": arange(0.1,10000.1,0.1),
#                     "spike_weights": poiss*0.369}])

poiss_delay = 0.2
poiss_weight = 0.369
nest.CopyModel("static_synapse", "poisson_connection",
               {"weight": poiss_weight, "delay": poiss_delay})

nest.Connect(pg, nodes_ex, {'rule': 'all_to_all'}, "poisson_connection")
nest.Connect(pg, nodes_in, {'rule': 'all_to_all'}, "poisson_connection")
#nest.Connect(sg, nodes_ex, {'rule': 'all_to_all'}, {'delay': poiss_delay})
#nest.Connect(sg, nodes_in, {'rule': 'all_to_all'}, {'delay': poiss_delay})

conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"], "interval":0.1})
nest.Connect(multimeter, [nodes_ex[0]])

build_time = time.time()

print("Simulating ...")

nest.Simulate(sim_time)

end_time = time.time()

print("Building time     : %.2f s" % (build_time - start_time))
print("Simulation time   : %.2f s" % (end_time - build_time))

dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

import numpy as np
np.savetxt('brunel_nest.dat',np.transpose([ts, Vms]))

#import pylab
#pylab.figure(1)
#pylab.plot(ts, Vms)

