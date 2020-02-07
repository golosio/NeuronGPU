import neuralgpu as ngpu


sg = ngpu.Create("spike_generator",2)
parrot = ngpu.Create("parrot_neuron")
parrot2 = ngpu.Create("parrot_neuron")
sg0=sg[0:0]
sg1=sg[1:1]

t0 = 50.0
dt = 1.0

spike_time0 = [t0]
spike_time1 = [t0+dt]
spike_height = [1.0]
n_spikes = 1

# set spike times and height
ngpu.SetStatus(sg0, {"spike_time": spike_time0, "spike_height":spike_height})
ngpu.SetStatus(sg1, {"spike_time": spike_time1, "spike_height":spike_height})
delay = 1.0
weight0 = 50.0
weight1 = 1.0

conn_dict={"rule": "one_to_one"}
syn_dict0={"weight":weight0, "delay":delay, "receptor":0, "synapse_group":1}
syn_dict1={"weight":weight1, "delay":delay, "receptor":0, "synapse_group":0}

ngpu.Connect(sg0, parrot, conn_dict, syn_dict0)
ngpu.Connect(sg1, parrot, conn_dict, syn_dict1)
ngpu.Connect(parrot, parrot2, conn_dict, syn_dict1)

conn_id = ngpu.GetConnections(sg0, parrot)
conn_status_dict = ngpu.GetStatus(conn_id, ["weight", "delay"])
print (conn_status_dict[0])

ngpu.Simulate(100.0)

conn_status_dict = ngpu.GetStatus(conn_id, ["weight", "delay"])
print (conn_status_dict[0])
