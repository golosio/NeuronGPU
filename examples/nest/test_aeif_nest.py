import pylab
import nest
nest.ResetKernel()

neuron = nest.Create("aeif_cond_alpha")
#neuron = nest.Create("iaf_neuron")

nest.SetStatus(neuron, {"I_e": 700.0})

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"],
                            "interval":0.1})

nest.Connect(multimeter, neuron)

nest.Simulate(2500.0)

dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)

import numpy as np
np.savetxt('aeif_nest.dat',np.transpose([ts, Vms]))
