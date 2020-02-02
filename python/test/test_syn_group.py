import neuralgpu as ngpu

parrot = ngpu.Create("parrot_neuron")

spike = ngpu.Create("spike_generator")
spike_time = [50.0, 400.0]
spike_height = [1.0, 1.0]
n_spikes = 2

# set spike times and height
ngpu.SetStatus(spike, {"spike_time": spike_time, "spike_height":spike_height})
delay = [50.0, 100.0, 150.0]
weight = [1.0, 1.0, 1.0]

conn_spec={"rule": "all_to_all"}
for syn in range(3):
    syn_spec={ 'synapse_group': syn,
              'receptor': 0, 'weight': weight[syn], 'delay': delay[syn]}
    ngpu.Connect(spike, parrot, conn_spec, syn_spec)

record = ngpu.CreateRecord("", ["V"], [parrot[0]], [0]);

ngpu.Simulate(800.0)

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
