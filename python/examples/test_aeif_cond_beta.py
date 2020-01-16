import sys
import neuralgpu as ngpu

neuron = ngpu.Create('aeif_cond_beta', 1, 3)
ngpu.SetStatus(neuron, {"V_peak": 0.0, "a": 4.0, "b":80.5,
                        "E_L":-70.6, "g_L":300.0})
ngpu.SetStatus(neuron, {'E_rev':[20.0, 0.0, -85.0],
                        'taus_decay':[40.0, 20.0, 30.0],
                        'taus_rise':[20.0, 10.0, 5.0]})
spike = ngpu.CreateSpikeGenerator(1)
spike_time = [10.0]
spike_height = [1.0]
n_spikes = 1
sg_node = 0 # this spike generator has only one node
# set spike times and height
ngpu.SetSpikeGenerator(0, spike_time, spike_height)
delay = [1.0, 100.0, 130.0]
weight = [0.1, 0.2, 0.15]

conn_spec={"rule": "all_to_all"}
for syn in range(3):
    syn_spec={ #'model': 'static_synapse', 'receptor_type': syn,
              'receptor': syn, 'weight': weight[syn], 'delay': delay[syn]}
    ngpu.Connect(spike, neuron, conn_spec, syn_spec)

record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

ngpu.Simulate(300.0)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V_m=[row[1] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V_m)

plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close()
