import sys
import neuralgpu as ngpu

dt_step = 5.0
N = 50

syn_group = ngpu.CreateSynGroup("stdp")

sg = ngpu.Create("spike_generator", N)
neuron = ngpu.Create("aeif_cond_beta", 2*N, 2)
ngpu.SetStatus(neuron, {"n_refractory_steps": 10000.0})
ngpu.SetStatus(neuron, {"taus_rise":[2.0, 1000.0], "taus_decay":[20.0, 2000]})
neuron0 = neuron[0:N-1]
neuron1 = neuron[N:2*N-1]
dt_list = []
for i in range(N):
    dt_list.append(dt_step*(-0.5*(N-1) + i))

spike_time = [50.0]
spike_height = [1.0]
n_spikes = 1
time_diff = 400.0

# set spike times and height
ngpu.SetStatus(sg, {"spike_time": spike_time, "spike_height":spike_height})
delay0 = 1.0
delay1 = delay0 + time_diff
weight_sg = 17.9
weight_stdp = 50.0

conn_dict={"rule": "one_to_one"}
syn_dict0={"weight":weight_sg, "delay":delay0, "receptor":0, "synapse_group":0}
syn_dict1={"weight":weight_sg, "delay":delay1, "receptor":0, "synapse_group":0}

ngpu.Connect(sg, neuron0, conn_dict, syn_dict0)
ngpu.Connect(sg, neuron1, conn_dict, syn_dict1)

for i in range(N):
    delay_stdp = time_diff - dt_list[i]
    syn_dict_stdp={"weight":weight_stdp, "delay":delay_stdp, "receptor":1, \
                   "synapse_group":syn_group}
    ngpu.Connect([neuron0[i]], [neuron1[i]], conn_dict, syn_dict_stdp)

ngpu.Simulate(200.0)

conn_id = ngpu.GetConnections(neuron0, neuron1)
conn_status_dict = ngpu.GetStatus(conn_id, ["weight", "delay"])
#print (conn_status_dict)
for i in range(N):
    print dt_list[i], conn_status_dict[i][0]
    #expect_w = dt_list[i]*fact + offset
    #if abs(expect_w - conn_status_dict[i][0])>tolerance:
    #    print("Expected weight: ", expect_w, " simulated: ", \
    #          conn_status_dict[i][0])
    #    sys.exit(1)

#sys.exit(0)
