import neuralgpu as ngpu

neuron = ngpu.CreateNeuron("AEIF", 1, 1)

ngpu.SetNeuronParam("I_e", neuron, 1000.0)

record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

ngpu.Simulate()

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
