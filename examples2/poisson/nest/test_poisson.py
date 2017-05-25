import pylab
import nest
nest.SetKernelStatus({"grng_seed" : 12345})
nest.SetKernelStatus({"rng_seeds" : [12346]})
neuron = nest.Create("aeif_cond_beta_multisynapse")
nest.SetStatus(neuron, {"tau_rise": [2.0], "tau_decay": [20.0]})


pg = nest.Create("poisson_generator")
nest.SetStatus(pg, {"rate": 1000.0})

syn_dict = {"weight": 1.0, "delay": 0.2, "receptor_type":1}

nest.Connect([pg[0]], neuron, syn_spec=syn_dict)

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
nest.Connect(multimeter, neuron)

nest.Simulate(1000.0)

dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)


pylab.show()
